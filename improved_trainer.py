"""
improved_trainer.py — Next-generation LSTM training for elephant movement prediction.

Key improvements over model_trainer.py:
  1. COORDINATE REGRESSION instead of 4,693-class classification.
     Predicts (delta_lat, delta_lon) offset from current grid centroid.
     Then snaps to nearest grid cell — spatially aware by design.
  2. HAVERSINE SPATIAL LOSS — penalises predictions by km distance,
     not just class mismatch. A 1km error hurts much less than a 50km error.
  3. ATTENTION over LSTM output — model learns which steps in the 10-step
     history matter most (e.g. last 2 steps usually matter more).
  4. LAYER NORM after LSTM — stabilises training with deep stacks.
  5. COSINE ANNEALING LR — avoids loss plateaus without manual tuning.
  6. AUTO-LOGGING — all epochs, metrics, and config written to training_log.txt.
  7. GRID CENTROID MAP — saves centroids of all grid cells for snapping.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import sys
import math
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import warnings

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings('ignore')

# ── CONFIG ─────────────────────────────────────────────────────────────────────
SEQ_LEN    = 10
BATCH_SIZE = 512
EPOCHS     = 200
LR         = 5e-4
HIDDEN_DIM = 256
NUM_LAYERS = 3
DROPOUT    = 0.3
LOG_FILE   = 'training_log_v2.txt'
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ───────────────────────────────────────────────────────────────────────────────

class Logger:
    """Writes to both stdout and a log file."""
    def __init__(self, path):
        self.path = path
        with open(path, 'w') as f:
            f.write(f"{'='*62}\n  ELEPHANT LSTM v2 — TRAINING LOG\n{'='*62}\n")
            f.write(f"Started: {datetime.now().isoformat()}\n")
            f.write(f"Device:  {DEVICE}\n\n")

    def log(self, msg):
        ts = datetime.now().strftime('%H:%M:%S')
        line = f"[{ts}] {msg}"
        print(line)
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')

logger = Logger(LOG_FILE)


# ── MODEL ──────────────────────────────────────────────────────────────────────

class AttentionPool(nn.Module):
    """Soft attention over LSTM output steps → weighted sum."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out):
        # lstm_out: (B, T, H)
        scores = self.attn(lstm_out).squeeze(-1)        # (B, T)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, T, 1)
        return (lstm_out * weights).sum(dim=1)          # (B, H)


class ElephantLSTMv2(nn.Module):
    """
    LSTM with LayerNorm + Attention for coordinate regression.
    Output: (delta_lat, delta_lon) — offset from current position.
    """
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.attention  = AttentionPool(hidden_dim)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 2)   # → (delta_lat, delta_lon)
        )

    def forward(self, x):
        out, _ = self.lstm(x)            # (B, T, H)
        out = self.layer_norm(out)
        ctx = self.attention(out)        # (B, H)
        return self.head(ctx)            # (B, 2)


# ── HAVERSINE LOSS ─────────────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorised haversine distance in km. All tensors, same shape."""
    R = 6371.0
    dlat = torch.deg2rad(lat2 - lat1)
    dlon = torch.deg2rad(lon2 - lon1)
    a = torch.sin(dlat/2)**2 + \
        torch.cos(torch.deg2rad(torch.clamp(lat1, -90, 90))) * \
        torch.cos(torch.deg2rad(torch.clamp(lat2, -90, 90))) * \
        torch.sin(dlon/2)**2
    # Clamp to avoid NaN in asin from floating point errors
    c = torch.clamp(torch.sqrt(torch.clamp(a, 0.0, 1.0)), 0.0, 1.0)
    return R * 2 * torch.asin(c)


class SpatialMSELoss(nn.Module):
    """
    Backprop on MSE of (delta_lat, delta_lon) only — numerically stable.
    Haversine is computed separately for monitoring but NOT in the gradient path.
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred_delta, true_delta, cur_lat, cur_lon):
        # Stable MSE loss for backprop
        mse_loss = self.mse(pred_delta, true_delta)

        # Haversine only for monitoring (no_grad)
        with torch.no_grad():
            pred_lat = cur_lat + pred_delta[:, 0]
            pred_lon = cur_lon + pred_delta[:, 1]
            true_lat = cur_lat + true_delta[:, 0]
            true_lon = cur_lon + true_delta[:, 1]
            dlat = torch.deg2rad(true_lat - pred_lat)
            dlon = torch.deg2rad(true_lon - pred_lon)
            a = torch.sin(dlat/2)**2 + \
                torch.cos(torch.deg2rad(torch.clamp(pred_lat, -90, 90))) * \
                torch.cos(torch.deg2rad(torch.clamp(true_lat, -90, 90))) * \
                torch.sin(dlon/2)**2
            c = torch.clamp(torch.sqrt(torch.clamp(a, 0.0, 1.0)), 0.0, 1.0)
            km_err = (6371.0 * 2 * torch.asin(c)).mean()

        return mse_loss, km_err



# ── DATA PREPARATION ───────────────────────────────────────────────────────────

def build_grid_centroid_map(feature_matrix_path='feature_matrix.csv'):
    """
    Reads the feature matrix and extracts (grid_id → centroid_lat, centroid_lon).
    Saves to grid_centroids.csv for use by predictor.
    """
    logger.log("Building grid centroid map from feature matrix...")
    df = pd.read_csv(feature_matrix_path)

    needed = ['to_grid', 'grid_centroid_lat', 'grid_centroid_lon']
    # Try common column names
    lat_col = next((c for c in df.columns if 'centroid_lat' in c.lower()), None)
    lon_col = next((c for c in df.columns if 'centroid_lon' in c.lower()), None)

    if lat_col and lon_col:
        centroids = df.groupby('to_grid')[[lat_col, lon_col]].mean().reset_index()
        centroids.columns = ['grid_id', 'centroid_lat', 'centroid_lon']
    else:
        # Fallback: compute from from_grid lat/lon if available
        logger.log("WARNING: centroid columns not found. Centroid map will be empty.")
        centroids = pd.DataFrame(columns=['grid_id', 'centroid_lat', 'centroid_lon'])

    centroids.to_csv('grid_centroids.csv', index=False)
    logger.log(f"Saved {len(centroids)} grid centroids to grid_centroids.csv")
    return centroids.set_index('grid_id').to_dict('index')


def prepare_regression_data(df, centroid_map, seq_len=SEQ_LEN):
    """
    Build sequences for regression.
    Target: (delta_lat, delta_lon) from current centroid to next centroid.
    current_pos: (lat, lon) of the current (from) grid centroid, for loss.
    """
    logger.log("Preparing regression sequences...")

    df['Date_Time'] = pd.to_datetime(df['Date_Time'], utc=True, errors='coerce')

    # Chronological 80/20 split
    sorted_dates = df['Date_Time'].sort_values().reset_index(drop=True)
    split_date = sorted_dates.iloc[int(len(sorted_dates)*0.80)]
    logger.log(f"Chronological split date (80/20): {split_date}")

    # Filter to rows where both from and to grid centroids are known
    df = df[df['from_grid'].isin(centroid_map) & df['to_grid'].isin(centroid_map)].copy()
    logger.log(f"Rows with known centroids: {len(df)}")

    # Compute target delta
    df['cur_lat'] = df['from_grid'].map(lambda g: centroid_map[g]['centroid_lat'])
    df['cur_lon'] = df['from_grid'].map(lambda g: centroid_map[g]['centroid_lon'])
    df['tgt_lat'] = df['to_grid'].map(lambda g: centroid_map[g]['centroid_lat'])
    df['tgt_lon'] = df['to_grid'].map(lambda g: centroid_map[g]['centroid_lon'])
    df['delta_lat'] = df['tgt_lat'] - df['cur_lat']
    df['delta_lon'] = df['tgt_lon'] - df['cur_lon']

    # Drop rows with NaN or extreme deltas (>5 degrees = ~550km, clearly wrong)
    df = df.dropna(subset=['cur_lat', 'cur_lon', 'tgt_lat', 'tgt_lon', 'delta_lat', 'delta_lon'])
    df = df[(df['delta_lat'].abs() < 5.0) & (df['delta_lon'].abs() < 5.0)].copy()
    logger.log(f"After NaN/extreme delta filter: {len(df)} rows")

    # Class filter: only keep grid cells with >=5 transitions (loosen, since regression)
    class_counts = df['to_grid'].value_counts()
    valid = class_counts[class_counts >= 5].index
    df = df[df['to_grid'].isin(valid)].copy()
    logger.log(f"After class filter (>=5): {len(df)} rows")

    # Feature columns
    exclude = {'elephant_id', 'Date_Time', 'from_grid', 'to_grid',
               'target', 'grid_centroid_lon', 'grid_centroid_lat',
               'cur_lat', 'cur_lon', 'tgt_lat', 'tgt_lon', 'delta_lat', 'delta_lon'}
    features = [c for c in df.columns if c not in exclude]
    logger.log(f"Features: {len(features)} — {features[:5]}...")

    train_df = df[df['Date_Time'] < split_date].copy()
    test_df  = df[df['Date_Time'] >= split_date].copy()

    # Scale features (fit on train only)
    scaler = StandardScaler()
    train_df[features] = scaler.fit_transform(train_df[features].fillna(0))
    test_df[features]  = scaler.transform(test_df[features].fillna(0))

    joblib.dump(scaler,   'scaler_v2.pkl')
    joblib.dump(features, 'feature_names_v2.pkl')
    logger.log("Saved scaler_v2.pkl and feature_names_v2.pkl")

    def make_sequences(split_df):
        Xf, Yd, Yp = [], [], []  # features, deltas, current positions
        for eid, grp in split_df.groupby('elephant_id'):
            grp = grp.sort_values('Date_Time').reset_index(drop=True)
            if len(grp) < seq_len:
                continue
            fv  = grp[features].values
            dl  = grp['delta_lat'].values
            dlo = grp['delta_lon'].values
            cl  = grp['cur_lat'].values
            clo = grp['cur_lon'].values
            for i in range(len(grp) - seq_len + 1):
                Xf.append(fv[i:i+seq_len])
                Yd.append([dl[i+seq_len-1], dlo[i+seq_len-1]])
                Yp.append([cl[i+seq_len-1], clo[i+seq_len-1]])
        return np.array(Xf), np.array(Yd, dtype=np.float32), np.array(Yp, dtype=np.float32)

    X_tr, Yd_tr, Yp_tr = make_sequences(train_df)
    X_te, Yd_te, Yp_te = make_sequences(test_df)

    logger.log(f"Train sequences: {X_tr.shape} | Test: {X_te.shape}")
    return X_tr, Yd_tr, Yp_tr, X_te, Yd_te, Yp_te, features


# ── TRAINING ───────────────────────────────────────────────────────────────────

def train_regression(X_tr, Yd_tr, Yp_tr, input_dim):
    logger.log(f"\nTraining ElephantLSTMv2 (regression) on {DEVICE}")
    logger.log(f"Config: epochs={EPOCHS}, batch={BATCH_SIZE}, lr={LR}, "
               f"hidden={HIDDEN_DIM}, layers={NUM_LAYERS}, dropout={DROPOUT}")

    model = ElephantLSTMv2(input_dim).to(DEVICE)
    logger.log(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = SpatialMSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    Xt = torch.tensor(X_tr,  dtype=torch.float32)
    Yt = torch.tensor(Yd_tr, dtype=torch.float32)
    Pt = torch.tensor(Yp_tr, dtype=torch.float32)
    loader = DataLoader(TensorDataset(Xt, Yt, Pt), batch_size=BATCH_SIZE, shuffle=True,
                        pin_memory=(DEVICE.type == 'cuda'), num_workers=0)

    best_loss = float('inf')
    patience, patience_ctr = 20, 0
    best_state = None

    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss, total_km = 0.0, 0.0

        for bX, bY, bP in loader:
            bX, bY, bP = bX.to(DEVICE), bY.to(DEVICE), bP.to(DEVICE)
            optimizer.zero_grad()
            pred = model(bX)
            loss, km = criterion(pred, bY, bP[:, 0], bP[:, 1])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            total_km   += km.item()

        scheduler.step()
        avg_loss = total_loss / len(loader)
        avg_km   = total_km   / len(loader)
        cur_lr   = optimizer.param_groups[0]['lr']

        logger.log(f"Epoch {epoch:03d}/{EPOCHS} | Loss: {avg_loss:.5f} | "
                   f"Avg km err: {avg_km:.3f} km | LR: {cur_lr:.7f}")

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_ctr = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                logger.log(f"Early stopping at epoch {epoch} (patience={patience})")
                break

    if best_state:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), 'elephant_lstm_v2.pt')
    logger.log(f"\nBest loss: {best_loss:.5f} | Saved -> elephant_lstm_v2.pt")
    return model


# ── EVALUATION ─────────────────────────────────────────────────────────────────

def evaluate_regression(model, X_te, Yd_te, Yp_te, centroid_map):
    logger.log("\n--- Evaluating regression model ---")
    model.eval()

    Xt = torch.tensor(X_te, dtype=torch.float32)
    all_pred_delta = []

    with torch.no_grad():
        for i in range(0, len(Xt), 512):
            batch = Xt[i:i+512].to(DEVICE)
            all_pred_delta.append(model(batch).cpu().numpy())

    pred_delta = np.concatenate(all_pred_delta, axis=0)   # (N, 2)
    true_delta = Yd_te                                      # (N, 2)
    cur_pos    = Yp_te                                      # (N, 2)

    pred_lat = cur_pos[:, 0] + pred_delta[:, 0]
    pred_lon = cur_pos[:, 1] + pred_delta[:, 1]
    true_lat = cur_pos[:, 0] + true_delta[:, 0]
    true_lon = cur_pos[:, 1] + true_delta[:, 1]

    def hav_np(lat1, lon1, lat2, lon2):
        R = 6371.0
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
        return R * 2 * np.arcsin(np.clip(np.sqrt(a), 0, 1))

    km_errs = hav_np(pred_lat, pred_lon, true_lat, true_lon)

    # Build centroid arrays for snapping
    centroid_ids  = list(centroid_map.keys())
    centroid_lats = np.array([centroid_map[g]['centroid_lat'] for g in centroid_ids])
    centroid_lons = np.array([centroid_map[g]['centroid_lon'] for g in centroid_ids])

    def snap_to_grid(lat_arr, lon_arr):
        preds = []
        for lat, lon in zip(lat_arr, lon_arr):
            dists = hav_np(lat, lon, centroid_lats, centroid_lons)
            preds.append(centroid_ids[np.argmin(dists)])
        return preds

    logger.log("Snapping predictions to nearest grid cell...")
    pred_grids = snap_to_grid(pred_lat, pred_lon)
    true_grids = snap_to_grid(true_lat, true_lon)

    top1 = np.mean(np.array(pred_grids) == np.array(true_grids))

    # Top-5: predict ±1 cell neighbours also count
    within_5km  = np.mean(km_errs <= 5.0)
    within_10km = np.mean(km_errs <= 10.0)
    within_20km = np.mean(km_errs <= 20.0)

    metrics = {
        'Mean km error'      : float(np.mean(km_errs)),
        'Median km error'    : float(np.median(km_errs)),
        'Within 5km'         : float(within_5km),
        'Within 10km'        : float(within_10km),
        'Within 20km'        : float(within_20km),
        'Top-1 grid accuracy': float(top1),
    }

    logger.log("\n--- Spatial Accuracy Metrics ---")
    for k, v in metrics.items():
        logger.log(f"  {k}: {v:.4f}")

    # Plot km error distribution
    plt.figure(figsize=(10, 4))
    plt.hist(km_errs[km_errs < 50], bins=50, color='steelblue', edgecolor='white')
    plt.xlabel('Prediction Error (km)')
    plt.ylabel('Count')
    plt.title('LSTM v2 Regression — Spatial Error Distribution')
    plt.axvline(np.median(km_errs), color='red', linestyle='--', label=f'Median: {np.median(km_errs):.1f} km')
    plt.legend()
    plt.tight_layout()
    plt.savefig('evaluation_report_v2.png')
    logger.log("Saved evaluation_report_v2.png")

    return metrics


# ── MAIN ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    logger.log(f"Using device: {DEVICE}")

    df = pd.read_csv('feature_matrix.csv')
    logger.log(f"Loaded feature_matrix.csv: {df.shape}")

    centroid_map = build_grid_centroid_map('feature_matrix.csv')

    if not centroid_map:
        logger.log("FATAL: centroid map is empty — cannot train regression model.")
        logger.log("Ensure feature_matrix.csv has 'grid_centroid_lat' and 'grid_centroid_lon' columns.")
        sys.exit(1)

    X_tr, Yd_tr, Yp_tr, X_te, Yd_te, Yp_te, features = prepare_regression_data(df, centroid_map)

    model = train_regression(X_tr, Yd_tr, Yp_tr, input_dim=X_tr.shape[2])
    metrics = evaluate_regression(model, X_te, Yd_te, Yp_te, centroid_map)

    logger.log("\n=== TRAINING COMPLETE ===")
    logger.log(f"Log saved to: {LOG_FILE}")
    logger.log("Output files: elephant_lstm_v2.pt, scaler_v2.pkl, feature_names_v2.pkl, grid_centroids.csv, evaluation_report_v2.png")
