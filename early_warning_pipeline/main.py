import torch
import pandas as pd
import numpy as np
import geopandas as gpd
from datetime import datetime
import os
import json
import joblib

# Import all modules
from data_loader import load_gps_data, validate_spatial_bounds
from grid_builder import build_grid, assign_gps_to_grid, compute_transitions
from gee_extractor import batch_extract_features
from human_features import extract_osm_features, merge_human_features
from memory_features import compute_memory_features, compute_site_fidelity
from feature_matrix import build_feature_matrix, engineer_features
from model_trainer import prepare_train_test, train_lstm, evaluate_model, ElephantLSTM
from predictor import predict_next_grid
from alert_engine import identify_at_risk_villages, generate_alert_report, plot_prediction_map

# ─── REAL DATA PATHS (auto-resolved relative to this script) ───────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
GPS_PATH  = os.path.join(BASE, 'doi_10_5061_dryad_dr7sqv9v9__v20200116', 'ElephantsData_ano.csv')
OSM_DIR   = os.path.join(BASE, 'botswana-260310-free.shp')
OSM_PLACES = os.path.join(OSM_DIR, 'gis_osm_places_free_1.shp')
OSM_ROADS  = os.path.join(OSM_DIR, 'gis_osm_roads_free_1.shp')
OSM_LAND   = os.path.join(OSM_DIR, 'gis_osm_landuse_a_free_1.shp')

# Botswana study area bounding box in EPSG:4326 — will be converted for grid
# Using the GPS data itself as the bounding reference
STUDY_LON_MIN, STUDY_LON_MAX = 23.0, 31.0
STUDY_LAT_MIN, STUDY_LAT_MAX = -23.0, -17.0

def main():
    print(f"[{datetime.now().isoformat()}] === STARTING ELEPHANT EARLY WARNING PIPELINE ===")

    # ── STAGE 1: LOAD & VALIDATE ──────────────────────────────────────────────
    print(f"\n[{datetime.now().isoformat()}] -- STAGE 1: Loading GPS Data --")
    gps_df = load_gps_data(GPS_PATH)
    
    # The Dryad CSV has no lon/lat columns — movement is encoded as Dist/Angle.
    # For the spatial grid we need coordinates. Since Dryad is anonymous,
    # we synthesize coordinates from Dist/Angle + known Botswana centroid.
    # Starting position: center of Chobe NP, the study area centroid.
    if 'Longitude' not in gps_df.columns and 'location-long' not in gps_df.columns:
        print(f"[{datetime.now().isoformat()}] No lon/lat columns found — synthesizing from Dist/Angle (dead reckoning from Botswana centroid)...")
        import numpy as np
        # Reconstruct tracks per elephant using dead reckoning
        START_LON, START_LAT = 25.0, -18.0  # Chobe NP area centroid
        DEG_PER_M = 1.0 / 111000.0           # approx meters → degrees latitude
        
        all_lons, all_lats = [], []
        for eid, grp in gps_df.groupby('id'):
            grp = grp.sort_values('Date_Time')
            lons = [START_LON]
            lats = [START_LAT]
            for i in range(1, len(grp)):
                dist_m = grp.iloc[i]['Dist']
                angle_rad = grp.iloc[i]['Angle']
                # angle from dryad is turning angle — accumulate heading
                heading = sum(grp.iloc[1:i+1]['Angle'].values)
                dx = dist_m * np.sin(heading) * DEG_PER_M
                dy = dist_m * np.cos(heading) * DEG_PER_M
                lons.append(np.clip(lons[-1] + dx, STUDY_LON_MIN, STUDY_LON_MAX))
                lats.append(np.clip(lats[-1] + dy, STUDY_LAT_MIN, STUDY_LAT_MAX))
            all_lons.extend(lons)
            all_lats.extend(lats)
            
        gps_df = gps_df.sort_values(['id', 'Date_Time']).reset_index(drop=True)
        gps_df['Longitude'] = all_lons
        gps_df['Latitude']  = all_lats
        print(f"[{datetime.now().isoformat()}] Dead-reckoning complete.")

    gps_df = validate_spatial_bounds(gps_df,
                                     lon_min=STUDY_LON_MIN, lon_max=STUDY_LON_MAX,
                                     lat_min=STUDY_LAT_MIN, lat_max=STUDY_LAT_MAX)

    # ── STAGE 2: BUILD GRID ───────────────────────────────────────────────────
    print(f"\n[{datetime.now().isoformat()}] -- STAGE 2: Building Spatial Grid --")
    import pyproj
    from shapely.ops import transform as shapely_transform
    from shapely.geometry import box
    import functools

    # Project study bbox to UTM 34S to get metre-space bounds
    proj_wgs = pyproj.CRS('EPSG:4326')
    proj_utm = pyproj.CRS('EPSG:32734')
    transformer = pyproj.Transformer.from_crs(proj_wgs, proj_utm, always_xy=True)
    
    min_x, min_y = transformer.transform(STUDY_LON_MIN, STUDY_LAT_MIN)
    max_x, max_y = transformer.transform(STUDY_LON_MAX, STUDY_LAT_MAX)
    bounds_utm = (min_x, min_y, max_x, max_y)

    grid_gdf = build_grid(bounds_utm, cell_size_m=5000)  # 5km cells — consistent with demo_scenarios
    gps_df   = assign_gps_to_grid(gps_df, grid_gdf)
    transitions_df = compute_transitions(gps_df)
    print(f"[{datetime.now().isoformat()}] Transitions computed: {len(transitions_df)} rows.")

    # ── STAGE 3: SATELLITE FEATURES (GEE — skipped if not authenticated) ─────
    print(f"\n[{datetime.now().isoformat()}] -- STAGE 3: Satellite Features --")
    gps_df = batch_extract_features(gps_df)

    # ── STAGE 4: HUMAN FEATURES (OSM) ─────────────────────────────────────────
    print(f"\n[{datetime.now().isoformat()}] -- STAGE 4: Human/OSM Features --")
    # We pass the places shapefile — human_features.py will detect fclass column
    grid_gdf = extract_osm_features(OSM_PLACES, grid_gdf)
    gps_df   = merge_human_features(gps_df, grid_gdf)

    # ── STAGE 5: MEMORY FEATURES ──────────────────────────────────────────────
    print(f"\n[{datetime.now().isoformat()}] -- STAGE 5: Memory Features --")
    transitions_df = compute_memory_features(transitions_df)
    gps_df         = compute_site_fidelity(gps_df)

    # ── STAGE 6: FEATURE MATRIX ───────────────────────────────────────────────
    print(f"\n[{datetime.now().isoformat()}] -- STAGE 6: Feature Matrix --")
    df = build_feature_matrix(gps_df, transitions_df, grid_gdf)
    df = engineer_features(df)
    out_csv = f'feature_matrix.csv'
    df.to_csv(out_csv, index=False)
    print(f"[{datetime.now().isoformat()}] Saved feature matrix ({df.shape}) to {out_csv}")

    # ── STAGE 7: LSTM MODEL TRAINING ──────────────────────────────────────────
    print(f"\n[{datetime.now().isoformat()}] -- STAGE 7: LSTM Model Training --")
    X_train, X_test, y_train, y_test = prepare_train_test(df)
    
    num_classes = len(np.unique(np.concatenate((y_train, y_test))))
    lstm_model = train_lstm(X_train, y_train, input_dim=X_train.shape[2], output_dim=num_classes, epochs=100, batch_size=256)
    metrics = evaluate_model(lstm_model, X_test, y_test)
    print("\n=== EVALUATION METRICS ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    scaler        = joblib.load('scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')

    # ── STAGE 8: LIVE PREDICTION DEMO ────────────────────────────────────────
    print(f"\n[{datetime.now().isoformat()}] -- STAGE 8: Live Prediction Demo --")
    # Load LSTM model for prediction
    features = joblib.load('feature_names.pkl')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pred_model = ElephantLSTM(input_dim=len(features), hidden_dim=128, num_layers=2, output_dim=num_classes)
    pred_model.load_state_dict(torch.load('elephant_lstm.pt', map_location=device))
    pred_model.to(device)
    pred_model.eval()
    
    # Build a sequence from the last 10 transition rows of elephant 1
    eid = transitions_df['elephant_id'].iloc[0]
    demo_seq = transitions_df[transitions_df['elephant_id'] == eid].sort_values('Date_Time').tail(10).to_dict('records')
    
    predictions, inference_context = predict_next_grid(pred_model, scaler, label_encoder, demo_seq)
    print("\nTop-5 Predicted Next Grid Cells:")
    print(predictions.to_string(index=False))

    # ── STAGE 9: ALERT ENGINE ─────────────────────────────────────────────────
    print(f"\n[{datetime.now().isoformat()}] -- STAGE 9: Alert Engine --")
    osm_villages_gdf = gpd.read_file(OSM_PLACES)
    villages_at_risk = identify_at_risk_villages(predictions, osm_villages_gdf)
    
    alert = generate_alert_report(predictions, villages_at_risk)
    last_step = demo_seq[-1]
    alert['elephant_id'] = last_step.get('elephant_id', 'Unknown')
    alert['current_grid'] = last_step.get('from_grid', 'Unknown')
    with open('alert_output.json', 'w') as f:
        json.dump(alert, f, indent=4)
    
    print("\n=== ALERT ===")
    print(json.dumps(alert, indent=2))

    plot_prediction_map(last_step.get('from_grid', ''), predictions, villages_at_risk, grid_gdf)

    print(f"\n[{datetime.now().isoformat()}] === PIPELINE COMPLETE ===")
    print("Output files: feature_matrix.csv, elephant_lstm.pt, evaluation_report.png, alert_output.json, alert_map.html")

if __name__ == '__main__':
    main()
