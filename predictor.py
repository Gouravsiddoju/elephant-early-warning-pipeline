"""
predictor.py — Unified predictor supporting both:
  - ElephantLSTMv2 (regression, coordinate offset) — used if elephant_lstm_v2.pt exists
  - ElephantLSTM   (classifier, 4693 classes)      — fallback to elephant_lstm.pt

The v2 regression model predicts (delta_lat, delta_lon) from the current
grid centroid, then snaps to the nearest grid cell.
"""
import pandas as pd
import numpy as np
import joblib
import torch
import os
from datetime import datetime

SEQ_LEN = 10

# ── LOAD MODEL ARCH ────────────────────────────────────────────────────────────
def _load_model(model_obj_or_path, scaler, label_encoder, version):
    """Internal helper — already-loaded model is passed in by callers."""
    return model_obj_or_path


def _hav_np(lat1, lon1, lat2, lon2):
    """Numpy haversine distance in km (arrays)."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.clip(np.sqrt(a), 0, 1))


def _build_input_tensor(current_input_seq, features, scaler):
    """Convert list-of-dicts sequence into scaled (1, SEQ_LEN, F) tensor."""
    input_df = pd.DataFrame(current_input_seq)

    # Derived features if missing
    if 'log_step_dist' in features and 'log_step_dist' not in input_df.columns and 'step_dist_m' in input_df.columns:
        input_df['log_step_dist'] = np.log1p(input_df['step_dist_m'].fillna(0))
    if 'cos_angle' in features and 'cos_angle' not in input_df.columns and 'turning_angle' in input_df.columns:
        input_df['cos_angle'] = np.cos(input_df['turning_angle'].fillna(0))
        input_df['sin_angle'] = np.sin(input_df['turning_angle'].fillna(0))
    if 'crop_attraction' in features and 'crop_attraction' not in input_df.columns and 'cropland_pct' in input_df.columns and 'ndvi' in input_df.columns:
        input_df['crop_attraction'] = input_df['cropland_pct'] * input_df['ndvi']
    if 'village_risk' in features and 'village_risk' not in input_df.columns and 'village_distance_m' in input_df.columns:
        input_df['village_risk'] = 1.0 / (input_df['village_distance_m'] + 1.0)

    for col in features:
        if col not in input_df.columns:
            input_df[col] = 0.0

    input_df = input_df[features].fillna(0.0)

    # Pad / trim to SEQ_LEN
    if len(input_df) < SEQ_LEN:
        pad = pd.DataFrame(np.zeros((SEQ_LEN - len(input_df), len(features))), columns=features)
        input_df = pd.concat([pad, input_df], ignore_index=True)
    elif len(input_df) > SEQ_LEN:
        input_df = input_df.tail(SEQ_LEN).reset_index(drop=True)

    X_scaled = scaler.transform(input_df.values)
    return torch.tensor([X_scaled], dtype=torch.float32)   # (1, SEQ_LEN, F)


def predict_next_grid(model, scaler, label_encoder, current_input_seq: list,
                      feature_names_path='feature_names.pkl',
                      centroids_path='grid_centroids.csv') -> pd.DataFrame:
    """
    Unified prediction function.
    Auto-detects v2 regression model vs v1 classifier from model class name.
    Returns top-5 DataFrame with grid_id, probability, centroid_lat, centroid_lon.
    """
    elephant_id = current_input_seq[-1].get('elephant_id', 'Unknown')
    model_class = type(model).__name__
    is_v2 = (model_class == 'ElephantLSTMv2')
    print(f"[{datetime.now().isoformat()}] Running prediction for {elephant_id} "
          f"using {'LSTM v2 (regression)' if is_v2 else 'LSTM v1 (classifier)'}...")

    # Load feature names
    try:
        feat_path = 'feature_names_v2.pkl' if is_v2 else feature_names_path
        features = joblib.load(feat_path)
    except Exception as e:
        print(f"Warning: Could not load features from {feat_path}. Using dict keys.")
        exclude = {'elephant_id', 'Date_Time', 'from_grid', 'to_grid', 'target',
                   'grid_centroid_lon', 'grid_centroid_lat'}
        features = [k for k in current_input_seq[-1].keys() if k not in exclude]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = model.to(device)
    model.eval()

    X_tensor = _build_input_tensor(current_input_seq, features, scaler).to(device)

    # Load centroid map (needed for both models)
    centroid_map = {}
    centroid_ids, centroid_lats, centroid_lons = [], [], []
    if os.path.exists(centroids_path):
        cd_df = pd.read_csv(centroids_path)
        centroid_map = cd_df.set_index('grid_id').to_dict('index')
        centroid_ids  = list(centroid_map.keys())
        centroid_lats = np.array([centroid_map[g]['centroid_lat'] for g in centroid_ids])
        centroid_lons = np.array([centroid_map[g]['centroid_lon'] for g in centroid_ids])

    results = []

    with torch.no_grad():
        output = model(X_tensor)

    if is_v2:
        # ── REGRESSION PATH ───────────────────────────────────────────────
        # output shape: (1, 2) → (delta_lat, delta_lon)
        delta = output.cpu().numpy()[0]  # [delta_lat, delta_lon]

        # Get current position from last step of sequence
        last_step = current_input_seq[-1]
        cur_lat = float(last_step.get('grid_centroid_lat') or last_step.get('cur_lat') or 0.0)
        cur_lon = float(last_step.get('grid_centroid_lon') or last_step.get('cur_lon') or 0.0)

        pred_lat = cur_lat + float(delta[0])
        pred_lon = cur_lon + float(delta[1])

        if centroid_ids:
            # Compute distances to all grid centroids and return top-5 nearest
            dists = _hav_np(pred_lat, pred_lon, centroid_lats, centroid_lons)
            top5_idx = np.argsort(dists)[:5]

            # Convert distances to pseudo-probabilities (softmax of negative distances)
            top5_dists = dists[top5_idx]
            weights = np.exp(-top5_dists / max(top5_dists[0], 0.5))  # softmax-like
            weights /= weights.sum()

            for rank, (idx, w) in enumerate(zip(top5_idx, weights), start=1):
                g = centroid_ids[idx]
                results.append({
                    'rank': rank,
                    'grid_id': g,
                    'probability': float(w),
                    'centroid_lat': float(centroid_lats[idx]),
                    'centroid_lon': float(centroid_lons[idx]),
                    'dist_km': float(dists[idx])
                })
        else:
            # No centroid map — just return raw prediction coordinate
            results.append({
                'rank': 1, 'grid_id': 'UNKNOWN',
                'probability': 1.0,
                'centroid_lat': pred_lat, 'centroid_lon': pred_lon, 'dist_km': 0.0
            })

    else:
        # ── CLASSIFIER PATH (v1 fallback) ─────────────────────────────────
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        top5_indices = np.argsort(probs)[-5:][::-1]
        top5_probs   = probs[top5_indices]
        top5_grids   = label_encoder.inverse_transform(top5_indices) if label_encoder else top5_indices

        for rank, (g_id, p) in enumerate(zip(top5_grids, top5_probs), start=1):
            lat = centroid_map.get(g_id, {}).get('centroid_lat', np.nan)
            lon = centroid_map.get(g_id, {}).get('centroid_lon', np.nan)
            results.append({
                'rank': rank, 'grid_id': g_id,
                'probability': float(p),
                'centroid_lat': float(lat), 'centroid_lon': float(lon)
            })

    res_df = pd.DataFrame(results)
    print(f"[{datetime.now().isoformat()}] Prediction complete. Top-1: {results[0]['grid_id']} "
          f"(p={results[0]['probability']:.3f})")
    return res_df


if __name__ == "__main__":
    pass
