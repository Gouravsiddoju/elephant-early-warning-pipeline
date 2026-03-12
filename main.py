import pandas as pd
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
from model_trainer import prepare_train_test, train_random_forest, evaluate_model
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
STUDY_LON_MIN, STUDY_LON_MAX = 23.0, 28.0
STUDY_LAT_MIN, STUDY_LAT_MAX = -22.0, -17.0

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

    grid_gdf = build_grid(bounds_utm, cell_size_m=2000)  # 2km cells — fewer classes, fits in RAM
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

    # ── STAGE 7: MODEL TRAINING ───────────────────────────────────────────────
    print(f"\n[{datetime.now().isoformat()}] -- STAGE 7: Model Training --")
    X_train, X_test, y_train, y_test = prepare_train_test(df)
    
    rf_model = train_random_forest(X_train, y_train)
    metrics  = evaluate_model(rf_model, X_test, y_test)
    print("\n=== EVALUATION METRICS ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    scaler        = joblib.load('scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')

    # ── STAGE 8: LIVE PREDICTION DEMO ────────────────────────────────────────
    print(f"\n[{datetime.now().isoformat()}] -- STAGE 8: Live Prediction Demo --")
    # Use stats from the real data as sensible demo inputs
    demo_input = {
        'elephant_id': '1',
        'from_grid': transitions_df.iloc[0]['from_grid'],
        'date': '2015-08-01',
        'step_dist_m': float(gps_df['Dist'].median()),
        'turning_angle': 0.0,
        'ndvi': 0.5,
        'rainfall_7d_mm': 5.0,
        'village_distance_m': gps_df.get('village_distance_m', pd.Series([5000.0])).median(),
        'cropland_pct': 0.05,
        'season': 2,
        'time_of_day': 1,
        'repeat_count': 2,
        'success_score': 0.4,
        'landcover_class': 9,
        'is_forest': 0,
        'is_cropland': 0,
        'visit_count': 5,
        'last_visit_days_ago': 10.0,
        'is_home_range_core': 0,
        'hour': 10,
        'month': 8,
        'Field': 0, 'River': 0, 'Corridor': 1, 'Trees': 0
    }

    predictions = predict_next_grid(rf_model, scaler, label_encoder, demo_input)
    print("\nTop-5 Predicted Next Grid Cells:")
    print(predictions.to_string(index=False))

    # ── STAGE 9: ALERT ENGINE ─────────────────────────────────────────────────
    print(f"\n[{datetime.now().isoformat()}] -- STAGE 9: Alert Engine --")
    osm_villages_gdf = gpd.read_file(OSM_PLACES)
    villages_at_risk = identify_at_risk_villages(predictions, osm_villages_gdf)
    
    alert = generate_alert_report(predictions, villages_at_risk)
    alert['elephant_id'] = demo_input['elephant_id']
    alert['current_grid'] = demo_input['from_grid']
    with open('alert_output.json', 'w') as f:
        json.dump(alert, f, indent=4)
    
    print("\n=== ALERT ===")
    print(json.dumps(alert, indent=2))

    plot_prediction_map(demo_input['from_grid'], predictions, villages_at_risk, grid_gdf)

    print(f"\n[{datetime.now().isoformat()}] === PIPELINE COMPLETE ===")
    print("Output files: feature_matrix.csv, elephant_model.pkl, evaluation_report.png, alert_output.json, alert_map.html")

if __name__ == '__main__':
    main()
