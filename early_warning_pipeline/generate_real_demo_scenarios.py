"""
generate_real_demo_scenarios.py
Generate 5 demo scenarios for the React dashboard using REAL elephant data from feature_matrix.csv.
Stages elephants in specific situations (Safe, Crop Raid, Proximity, Poaching, Boundary).

Output: dashboard/public/scenarios/scenario_{1-5}.json
"""

import os, sys, json, shutil, math, random
import torch, joblib, numpy as np, pandas as pd, pyproj
from datetime import datetime, timezone

# Add parent dir to path if needed to find local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
OUT_DIR       = os.path.normpath(os.path.join(BASE, '..', 'dashboard', 'public', 'scenarios'))
os.makedirs(OUT_DIR, exist_ok=True)

SEQ_LEN     = 10
N_ELEPHANTS = 8
MIN_HISTORY = SEQ_LEN
DEG_PER_M   = 1.0 / 111_000.0

_GRID_LAT_ORIGIN = -17.0 + 0.02244
_GRID_LON_ORIGIN = 23.0  + 0.02244
_GRID_LAT_STEP   = -0.04488
_GRID_LON_STEP   =  0.04488

# Important geographic anchors
VILLAGE_KASANE = (-17.87, 25.18)
VILLAGE_NATA   = (-19.52, 26.30)
VILLAGE_MAUN   = (-19.98, 23.42)
PK_BOUNDARY    = (-18.50, 27.20)

def resolve_latlon(grid_id, centroid_map):
    if grid_id in centroid_map:
        return float(centroid_map[grid_id]['centroid_lat']), \
               float(centroid_map[grid_id]['centroid_lon'])
    
    raise ValueError(f"Grid ID {grid_id} not found in spatial index. "
                     "This usually means the elephant is outside the trained study area.")

def dead_reckon_path(history_rows, start_lat, start_lon):
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
    print("=== Generating Real Demo Scenarios ===\n")

    # Load resources
    scaler        = joblib.load(SCALER_PKL)
    label_encoder = joblib.load(LE_PKL)
    features      = joblib.load(FEATNAMES_PKL)
    num_classes   = len(label_encoder.classes_)
    device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ElephantLSTM(input_dim=len(features), hidden_dim=128, num_layers=2, output_dim=num_classes)
    model.load_state_dict(torch.load(MODEL_PT, map_location=device))
    model.to(device); model.eval()

    # Load spatial index (CENTROIDS_CSV)
    if not os.path.exists(CENTROIDS_CSV) or os.path.getsize(CENTROIDS_CSV) < 1000:
        raise RuntimeError(f"FATAL: {CENTROIDS_CSV} is missing or an LFS pointer. Run: 'git lfs pull'")
    
    centroid_map = pd.read_csv(CENTROIDS_CSV).set_index('grid_id').to_dict('index')

    # Load data
    df = pd.read_csv(FEATURE_CSV)
    df['Date_Time'] = pd.to_datetime(df['Date_Time'], utc=True, errors='coerce')
    counts = df.groupby('elephant_id').size().sort_values(ascending=False)
    selected = counts[counts >= MIN_HISTORY].index[:N_ELEPHANTS].tolist()
    
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    exclude_cols = {'elephant_id','Date_Time','from_grid','to_grid','target','grid_centroid_lon','grid_centroid_lat'}

    # Scenario Definitions
    scenarios_meta = [
        {"id": 1, "title": "All Clear", "desc": "Standard operation. All elephants within park boundaries."},
        {"id": 2, "title": "Crop Raid Alert", "desc": "Elephant 5 and 9 spotted moving towards Nata farmlands."},
        {"id": 3, "title": "Village Proximity", "desc": "Elephant 7 and 11 within 1km of Kasane & Maun residential areas."},
        {"id": 4, "title": "Poaching Threat", "desc": "Erratic behavior and stampede patterns detected in 3 herds."},
        {"id": 5, "title": "Boundary Crossing", "desc": "Elephant 2 and 8 crossing into private concession NG/32."},
    ]

    for meta in scenarios_meta:
        sid = meta["id"]
        print(f"Generating Scenario {sid}: {meta['title']}...")

        elephants_out, preds_map, villages_map, move_map, history_map, alerts_out = [], {}, {}, {}, {}, []

        for eid in selected:
            eid_str = f"E-{eid}"
            grp = df[df['elephant_id'] == eid].sort_values('Date_Time')
            rows = grp.tail(25).copy().reset_index(drop=True)
            
            # --- Scenario Staging ---
            lat_off, lon_off = 0.0, 0.0
            status = "safe"
            custom_alert = None
            custom_villages = []

            if sid == 2: # Crop Raid
                if eid in [5, 9]:
                    lat_off, lon_off = VILLAGE_NATA[0] - rows.iloc[-1]['grid_centroid_lat'], VILLAGE_NATA[1] - rows.iloc[-1]['grid_centroid_lon']
                    lat_off += random.uniform(-0.02, 0.02); lon_off += random.uniform(-0.02, 0.02)
                    status = "danger"
                    custom_alert = f"⚠️ Elephant {eid} entering Nata farmlands — immediate crop risk"
                    custom_villages = [{"name": "Nata Farms", "distanceKm": 0.8, "population": 450, "atRisk": True}]
            elif sid == 3: # Village Proximity
                if eid in [7, 11]:
                    target = VILLAGE_KASANE if eid == 7 else VILLAGE_MAUN
                    lat_off, lon_off = target[0] - rows.iloc[-1]['grid_centroid_lat'], target[1] - rows.iloc[-1]['grid_centroid_lon']
                    lat_off += random.uniform(0.005, 0.01); lon_off += random.uniform(0.005, 0.01)
                    status = "danger"
                    custom_alert = f"🚨 Elephant {eid} is {1.1 if eid==7 else 0.9}km from settlement outskirts"
                    custom_villages = [{"name": "Residential Area", "distanceKm": 0.9, "population": 5200, "atRisk": True}]
            elif sid == 4: # Poaching
                if eid in [5, 3, 11]:
                    status = "danger"
                    custom_alert = f"🚨 Erratic movement detected for Elephant {eid} — possible poaching threat"
                    # Add jitter to movement for "erratic" look
                    rows['step_dist_m'] = rows['step_dist_m'] * random.uniform(2.0, 4.0)
            elif sid == 5: # Boundary
                if eid in [2, 8]:
                    lat_off, lon_off = PK_BOUNDARY[0] - rows.iloc[-1]['grid_centroid_lat'], PK_BOUNDARY[1] - rows.iloc[-1]['grid_centroid_lon']
                    status = "warning"
                    custom_alert = f"⚠️ Elephant {eid} crossing park boundary into NG/32 concession"

            # ─── REASONING SYNC ──────────────────────────────────────────────
            # If we applied a lat_off/lon_off, the environmental features in the
            # CSV are now "stale". We must manually override them to match 
            # the staged scenario so reasoning strings are logical.
            if lat_off != 0 or lon_off != 0:
                # Mock a low village distance if we staged near a village/boundary
                rows.loc[rows.index[-1], 'village_distance_m'] = random.uniform(500, 1800)
                rows.loc[rows.index[-1], 'cropland_pct'] = random.uniform(5.0, 15.0)
            
            # Diversify NDVI to avoid the "0.5" repetition
            rows['ndvi'] = rows['ndvi'] * random.uniform(0.8, 1.2)
            # ─────────────────────────────────────────────────────────────────

            # Apply offsets to lat/lon for history and prediction
            cur_grid_row = rows.iloc[-1]
            try:
                la, lo = resolve_latlon(cur_grid_row['from_grid'], centroid_map)
                cur_lat, cur_lon = (la + lat_off, lo + lon_off)
            except ValueError:
                print(f"  [Skip] Elephant {eid} is in unknown grid {cur_grid_row['from_grid']}")
                continue

            # Prediction
            seq_rows = grp.tail(SEQ_LEN).copy()
            # Ensure we keep 'from_grid' for spatial masking, even if not a model feature
            pred_feats = [c for c in grp.columns if c not in exclude_cols or c == 'from_grid']
            input_seq = seq_rows[pred_feats].to_dict('records')
            preds_df, context = predict_next_grid(model, scaler, label_encoder, input_seq, feature_names_path=FEATNAMES_PKL, centroids_path=CENTROIDS_CSV)
            
            # Helper for reasoning in scenarios (Sync with prediction_service.py)
            def get_reasoning(ctx, p):
                reasons = []
                ndvi = ctx.get('ndvi', 0)
                habitat_score = int(min(ndvi / 0.7, 1.0) * 100)
                cropland = ctx.get('cropland_pct', 0)
                v_dist = ctx.get('village_distance_m', 20000)
                speed = ctx.get('step_dist_m', 0)
                
                if ndvi > 0.45:
                    reasons.append(f"🌿 Habitat: {habitat_score}/100. High biomass (NDVI: {ndvi:.2f}) strongly attracts foraging behavior.")
                elif ndvi < 0.15:
                    reasons.append(f"🏜️ Habitat: {habitat_score}/100. Sparse vegetation corridor; movement likely transit-based.")
                else:
                    reasons.append(f"🌿 Habitat: {habitat_score}/100. Moderate foraging potential detected.")
                
                if cropland > 8.0:
                    crop_score = int(min(cropland/25, 1.0)*100)
                    reasons.append(f"🌽 Resource: {crop_score}/100. High-calorie crop attraction ({cropland:.1f}%). Elephant may be targeting farms.")
                
                if v_dist < 2000:
                    reasons.append(f"🚨 Conflict Risk: CRITICAL. Human settlement at {v_dist/1000.0:.1f} km. Immediate action required.")
                elif v_dist < 8000:
                    reasons.append(f"⚠️ Conflict Risk: MODERATE. Elephant is approaching human settlement boundaries ({v_dist/1000.0:.1f} km).")
                else:
                    reasons.append(f"🛡️ Safety: SECURE. Area is {v_dist/1000.0:.1f} km from known settlements.")
                
                if speed > 1500:
                    vel_score = int(min(speed/3500, 1.0)*100)
                    reasons.append(f"🏹 Velocity: {vel_score}/100. Active migration response indicated by sustained velocity.")
                
                if p > 0.4:
                    reasons.append(f"🎯 Prediction: {int(p*100)}/100. High spatial consistency with verified migration corridors.")
                
                if not reasons:
                    reasons.append("📍 Activity: 100/100 Stationary. Browsing or resting behavior detected.")
                return reasons

            preds_out = []
            for _, pr in preds_df.iterrows():
                try:
                    plat, plon = resolve_latlon(str(pr['grid_id']), centroid_map)
                    plat, plon = (plat + lat_off, plon + lon_off)
                except ValueError:
                    continue # Use only valid coordinate predicted cells
                preds_out.append({
                    "rank": int(pr['rank']), "gridCell": str(pr['grid_id']), "confidence": round(float(pr['probability'])*100, 2),
                    "location": {"lat": round(plat, 5), "lng": round(plon, 5)},
                    "distanceKm": round(((plat-cur_lat)**2 + (plon-cur_lon)**2)**0.5 * 111.0, 1),
                    "reasoning": get_reasoning(context, float(pr['probability']))
                })

            # History
            hpath = dead_reckon_path(rows, cur_lat, cur_lon)
            
            elephants_out.append({
                "id": eid_str, "name": f"Elephant {eid}", "gridCell": cur_grid_row['from_grid'],
                "position": {"lat": round(cur_lat, 5), "lng": round(cur_lon, 5)},
                "horizon": 48, "model": "LSTM (PyTorch)", "lastUpdated": now_str, "status": status
            })
            preds_map[eid_str] = preds_out
            villages_map[eid_str] = custom_villages
            history_map[eid_str] = hpath
            move_map[eid_str] = [{"hour": r['Date_Time'].strftime("%H:%M"), "distance": round(float(r.get('step_dist_m', 0))/1000.0, 3)} for _, r in rows.iterrows()]
            
            alerts_out.append({
                "id": f"alert_{sid}_{eid}", "timestamp": now_str, "type": "prediction" if status=="safe" else "proximity" if sid in [2,3] else "movement",
                "severity": "high" if status=="danger" else "medium" if status=="warning" else "low",
                "message": custom_alert or f"Elephant {eid} is in a safe designated area.",
                "elephantId": eid_str
            })

        out_data = {
            "generatedAt": now_str, "elephants": elephants_out, "predictionsMap": preds_map,
            "villagesMap": villages_map, "alertEvents": alerts_out, "movementMap": move_map, "historyMap": history_map
        }
        
        with open(os.path.join(OUT_DIR, f"scenario_{sid}.json"), 'w') as f:
            json.dump(out_data, f, indent=2)

    # Index
    index = [{"id": s["id"], "title": s["title"], "description": s["desc"], "file": f"scenario_{s['id']}.json"} for s in scenarios_meta]
    with open(os.path.join(OUT_DIR, "index.json"), 'w') as f:
        json.dump(index, f, indent=2)

    print(f"\n[OK] Generated 5 scenarios in {OUT_DIR}")

if __name__ == '__main__':
    main()
