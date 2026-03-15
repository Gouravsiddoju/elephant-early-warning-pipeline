"""
export_dashboard_data.py
Generate elephant_data.json for the React dashboard by running the real
LSTM predictions for all elephants in the feature_matrix.csv.

Output: early_warning_pipeline/dashboard_data.json
  (then copied to dashboard/public/elephant_data.json automatically)

Run from: early_warning_pipeline/
    python export_dashboard_data.py
"""

import os, sys, json, shutil
import torch, joblib, numpy as np, pandas as pd, pyproj
from datetime import datetime, timezone

from model_trainer import ElephantLSTM
from predictor import predict_next_grid
from grid_builder import build_grid

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE          = os.path.dirname(os.path.abspath(__file__))
FEATURE_CSV   = os.path.join(BASE, 'feature_matrix.csv')
MODEL_PT      = os.path.join(BASE, 'elephant_lstm.pt')
SCALER_PKL    = os.path.join(BASE, 'scaler.pkl')
LE_PKL        = os.path.join(BASE, 'label_encoder.pkl')
FEATNAMES_PKL = os.path.join(BASE, 'feature_names.pkl')
CENTROIDS_CSV = os.path.join(BASE, 'grid_centroids.csv')
OUT_JSON      = os.path.join(BASE, 'dashboard_data.json')
DASH_PUBLIC   = os.path.join(BASE, '..', 'dashboard', 'public', 'elephant_data.json')

SEQ_LEN     = 10
N_ELEPHANTS = 8
MIN_HISTORY = SEQ_LEN

# Grid math constants (derived from grid_centroids.csv step sizes)
_GRID_LAT_ORIGIN = -17.0 + 0.02244
_GRID_LON_ORIGIN = 23.0  + 0.02244
_GRID_LAT_STEP   = -0.04488
_GRID_LON_STEP   =  0.04488

def grid_id_to_latlon(grid_id: str):
    try:
        parts = grid_id.split('_')
        row = int(parts[0][1:])
        col = int(parts[1][1:])
        return float(_GRID_LAT_ORIGIN + row * _GRID_LAT_STEP), \
               float(_GRID_LON_ORIGIN + col * _GRID_LON_STEP)
    except Exception:
        return None, None

def resolve_latlon(grid_id, grid_wgs84, centroid_map):
    if grid_id in grid_wgs84.index:
        c = grid_wgs84.loc[grid_id].geometry.centroid
        return float(c.y), float(c.x)
    if grid_id in centroid_map:
        return float(centroid_map[grid_id]['centroid_lat']), \
               float(centroid_map[grid_id]['centroid_lon'])
    return grid_id_to_latlon(grid_id)

def label_status(top_confidence, villages):
    has_risk = any(v.get('atRisk') for v in villages)
    if has_risk:
        return "danger"
    if top_confidence < 10.0:
        return "warning"
    return "safe"

DEG_PER_M = 1.0 / 111_000.0

def dead_reckon_path(history_rows, start_lat, start_lon):
    """Reconstruct GPS waypoints from step_dist_m + turning_angle."""
    coords = [[round(start_lat, 5), round(start_lon, 5)]]
    heading = 0.0
    for _, row in history_rows.iterrows():
        dist_m = float(row.get('step_dist_m', 0) or 0)
        angle  = float(row.get('turning_angle', 0) or 0)
        heading += angle
        dlat = dist_m * np.cos(heading) * DEG_PER_M
        dlon = dist_m * np.sin(heading) * DEG_PER_M / max(np.cos(np.radians(coords[-1][0])), 0.01)
        coords.append([round(coords[-1][0] + dlat, 5), round(coords[-1][1] + dlon, 5)])
    return coords

def main():
    print("=== Elephant Dashboard Data Export ===\n")

    # Load model
    scaler        = joblib.load(SCALER_PKL)
    label_encoder = joblib.load(LE_PKL)
    features      = joblib.load(FEATNAMES_PKL)
    num_classes   = len(label_encoder.classes_)
    device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ElephantLSTM(input_dim=len(features), hidden_dim=128,
                         num_layers=2, output_dim=num_classes)
    model.load_state_dict(torch.load(MODEL_PT, map_location=device))
    model.to(device); model.eval()
    print(f"[OK] LSTM loaded — {num_classes} classes, {len(features)} features")

    # Build grid
    proj = pyproj.Transformer.from_crs('EPSG:4326','EPSG:32734',always_xy=True)
    min_x, min_y = proj.transform(23.0,-22.0)
    max_x, max_y = proj.transform(28.0,-17.0)
    grid_gdf  = build_grid((min_x,min_y,max_x,max_y), cell_size_m=5000)
    grid_wgs84 = grid_gdf.to_crs('EPSG:4326').set_index('grid_id')

    centroid_map = {}
    if os.path.exists(CENTROIDS_CSV):
        centroid_map = pd.read_csv(CENTROIDS_CSV).set_index('grid_id').to_dict('index')

    # Load feature matrix
    df = pd.read_csv(FEATURE_CSV)
    df['Date_Time'] = pd.to_datetime(df['Date_Time'], utc=True, errors='coerce')
    valid = df['to_grid'].value_counts()
    df = df[df['to_grid'].isin(valid[valid >= 10].index)].copy()
    print(f"[OK] Feature matrix: {len(df)} rows, {df['elephant_id'].nunique()} elephants")

    counts = df.groupby('elephant_id').size().sort_values(ascending=False)
    selected = counts[counts >= MIN_HISTORY].index[:N_ELEPHANTS].tolist()
    print(f"[OK] Selected {len(selected)} elephants: {selected}\n")

    exclude_cols = {'elephant_id','Date_Time','from_grid','to_grid',
                    'target','grid_centroid_lon','grid_centroid_lat'}
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    elephants_out   = []
    predictions_map = {}
    movement_map    = {}
    history_map     = {}
    alerts_out      = []

    for i, eid in enumerate(selected):
        eid_str = f"E-{eid}"
        grp = df[df['elephant_id'] == eid].sort_values('Date_Time')
        rows = grp.tail(25).copy().reset_index(drop=True)

        # Current grid (walk backwards for resolvable ID)
        all_grids = rows['from_grid'].tolist()
        cur_lat, cur_lon, cur_grid = None, None, None
        for g in reversed(all_grids):
            la, lo = resolve_latlon(g, grid_wgs84, centroid_map)
            if la is not None:
                cur_lat, cur_lon, cur_grid = la, lo, g
                break

        if cur_lat is None:
            print(f"  [SKIP] Elephant {eid} — no resolvable grid")
            continue

        # Sequence for LSTM
        seq_rows  = grp.tail(SEQ_LEN)
        input_seq = seq_rows[[c for c in grp.columns if c not in exclude_cols]].to_dict('records')

        # Run prediction
        try:
            preds_df = predict_next_grid(model, scaler, label_encoder, input_seq,
                                         feature_names_path=FEATNAMES_PKL,
                                         centroids_path=CENTROIDS_CSV)
        except Exception as e:
            print(f"  [WARN] Prediction failed for {eid}: {e}")
            continue

        # Build predictions list
        preds_out = []
        for _, pr in preds_df.iterrows():
            gid  = str(pr['grid_id'])
            prob = float(pr['probability']) * 100  # convert to percent
            plat, plon = resolve_latlon(gid, grid_wgs84, centroid_map)
            if plat is None:
                plat, plon = grid_id_to_latlon(gid)
            dist_deg = ((plat - cur_lat)**2 + (plon - cur_lon)**2)**0.5 if plat else 0
            dist_km  = round(dist_deg * 111.0, 1)
            preds_out.append({
                "rank":       int(pr['rank']),
                "gridCell":   gid,
                "confidence": round(prob, 2),
                "location":   {"lat": round(plat, 5) if plat else 0,
                               "lng": round(plon, 5) if plon else 0},
                "distanceKm": dist_km,
            })

        top_conf = preds_out[0]["confidence"] if preds_out else 0.0
        villages_here = []  # no live village data — real village NLP requires OSM
        status = label_status(top_conf, villages_here)

        # Movement activity: last 24 hourly step distances (step_dist_m col)
        move_rows = grp.tail(24).copy()
        move_data = []
        for _, mr in move_rows.iterrows():
            ts = mr['Date_Time']
            hour_label = ts.strftime("%H:%M") if not pd.isna(ts) else "--"
            dist_km_m = round(float(mr.get('step_dist_m', 0) or 0) / 1000.0, 3)
            move_data.append({"hour": hour_label, "distance": dist_km_m})

        # Elephant object (matches Elephant interface in mock-data.ts)
        elephants_out.append({
            "id":          eid_str,
            "name":        f"Elephant {eid}",
            "gridCell":    cur_grid,
            "position":    {"lat": round(cur_lat, 5), "lng": round(cur_lon, 5)},
            "horizon":     48,
            "model":       "LSTM (PyTorch)",
            "lastUpdated": now_str,
            "status":      status,
        })

        predictions_map[eid_str] = preds_out
        movement_map[eid_str]    = move_data

        # Historical path via dead reckoning
        start_la, start_lo = cur_lat, cur_lon
        for g in all_grids:
            la, lo = resolve_latlon(g, grid_wgs84, centroid_map)
            if la is not None:
                start_la, start_lo = la, lo
                break
        history_map[eid_str] = dead_reckon_path(rows, start_la, start_lo)

        # Generate one alert event per elephant
        action_msg = (
            f"Elephant {eid} — Top prediction: {preds_out[0]['gridCell']} "
            f"({top_conf:.1f}% confidence) | {preds_out[0]['distanceKm']} km away"
            if preds_out else f"Elephant {eid} — prediction unavailable"
        )
        sev = "high" if status == "danger" else "medium" if status == "warning" else "low"
        alerts_out.append({
            "id":         f"auto_{eid_str}",
            "timestamp":  now_str,
            "type":       "prediction",
            "severity":   sev,
            "message":    action_msg,
            "elephantId": eid_str,
        })

        top_grid = preds_out[0]['gridCell'] if preds_out else "N/A"
        print(f"  [{i+1}/{len(selected)}] Elephant {eid} -> {top_grid} ({top_conf:.1f}%)  [{status}]")

    output = {
        "generatedAt":    now_str,
        "elephants":      elephants_out,
        "predictionsMap": predictions_map,
        "villagesMap":    {e["id"]: [] for e in elephants_out},
        "alertEvents":    alerts_out,
        "movementMap":    movement_map,
        "historyMap":     history_map,
    }

    with open(OUT_JSON, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n[OK] Saved -> {OUT_JSON}")

    # Copy to dashboard/public/
    dash_public_path = os.path.normpath(DASH_PUBLIC)
    os.makedirs(os.path.dirname(dash_public_path), exist_ok=True)
    shutil.copy2(OUT_JSON, dash_public_path)
    print(f"[OK] Copied -> {dash_public_path}")

if __name__ == '__main__':
    main()
