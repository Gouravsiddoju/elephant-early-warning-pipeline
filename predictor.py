import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import os

def predict_next_grid(model, scaler, label_encoder, current_input: dict, feature_names_path='feature_names.pkl', centroids_path='grid_centroids.csv') -> pd.DataFrame:
    """
    Scale input with loaded scaler.
    Run model.predict_proba().
    Return top-5 grid cells with probability scores as DataFrame.
    """
    print(f"[{datetime.now().isoformat()}] Running prediction for {current_input.get('elephant_id', 'Unknown')}...")
    
    try:
        features = joblib.load(feature_names_path)
    except Exception as e:
        print(f"Warning: Could not load {feature_names_path}. Using dictionary keys. Error: {e}")
        features = list(current_input.keys())
        exclude = ['elephant_id', 'Date_Time', 'from_grid', 'to_grid', 'target', 'date', 'grid_centroid_lon', 'grid_centroid_lat']
        features = [f for f in features if f not in exclude]
        
    input_df = pd.DataFrame([current_input])
    
    if 'log_step_dist' in features and 'log_step_dist' not in input_df.columns and 'step_dist_m' in input_df.columns:
        input_df['log_step_dist'] = np.log1p(input_df['step_dist_m'])
        
    if 'cos_angle' in features and 'cos_angle' not in input_df.columns and 'turning_angle' in input_df.columns:
        input_df['cos_angle'] = np.cos(input_df['turning_angle'])
        input_df['sin_angle'] = np.sin(input_df['turning_angle'])
        
    if 'is_nighttime' in features and 'is_nighttime' not in input_df.columns and 'time_of_day' in input_df.columns:
        input_df['is_nighttime'] = (input_df['time_of_day'] == 2).astype(int)
        
    if 'crop_attraction' in features and 'crop_attraction' not in input_df.columns and 'cropland_pct' in input_df.columns and 'ndvi' in input_df.columns:
        input_df['crop_attraction'] = input_df['cropland_pct'] * input_df['ndvi']
        
    if 'village_risk' in features and 'village_risk' not in input_df.columns and 'village_distance_m' in input_df.columns:
        input_df['village_risk'] = 1.0 / (input_df['village_distance_m'] + 1.0)
        
    for missing in [f for f in features if f not in input_df.columns]:
        input_df[missing] = 0.0
        
    X = input_df[features].values
    X_scaled = scaler.transform(X)
    
    probs = model.predict_proba(X_scaled)[0]
    
    top5_indices = np.argsort(probs)[-5:][::-1]
    top5_probs = probs[top5_indices]
    
    top5_grid_ids = label_encoder.inverse_transform(top5_indices) if label_encoder else top5_indices
    
    # Load centroids map if available
    centroid_map = {}
    if os.path.exists(centroids_path):
        cd_df = pd.read_csv(centroids_path)
        centroid_map = cd_df.set_index('grid_id').to_dict('index')
    
    results = []
    
    for rank, (g_id, p) in enumerate(zip(top5_grid_ids, top5_probs), start=1):
        lon = centroid_map.get(g_id, {}).get('centroid_lon', np.nan)
        lat = centroid_map.get(g_id, {}).get('centroid_lat', np.nan)
        
        results.append({
            'rank': rank,
            'grid_id': g_id,
            'probability': float(p),
            'centroid_lon': float(lon),
            'centroid_lat': float(lat)
        })
        
    res_df = pd.DataFrame(results)
    print(f"[{datetime.now().isoformat()}] Prediction complete.")
    return res_df

if __name__ == "__main__":
    pass
