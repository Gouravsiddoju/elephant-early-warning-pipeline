import pandas as pd
import numpy as np
import joblib
import torch
import os
from datetime import datetime
from model_trainer import ElephantLSTM

SEQ_LEN = 10

def predict_next_grid(model, scaler, label_encoder, current_input_seq: list, feature_names_path='feature_names.pkl', centroids_path='grid_centroids.csv') -> pd.DataFrame:
    """
    Takes a sequence of inputs (list of dicts).
    Scale input with loaded scaler.
    Run PyTorch LSTM model.
    Return top-5 grid cells with probability scores as DataFrame.
    """
    elephant_id = current_input_seq[-1].get('elephant_id', 'Unknown')
    print(f"[{datetime.now().isoformat()}] Running prediction for {elephant_id} using PyTorch LSTM...")
    
    try:
        features = joblib.load(feature_names_path)
    except Exception as e:
        print(f"Warning: Could not load {feature_names_path}. Using dictionary keys. Error: {e}")
        features = list(current_input_seq[-1].keys())
        exclude = ['elephant_id', 'Date_Time', 'from_grid', 'to_grid', 'target', 'date', 'grid_centroid_lon', 'grid_centroid_lat']
        features = [f for f in features if f not in exclude]
        
    input_df = pd.DataFrame(current_input_seq)
    
    if 'log_step_dist' in features and 'log_step_dist' not in input_df.columns and 'step_dist_m' in input_df.columns:
        input_df['log_step_dist'] = np.log1p(input_df['step_dist_m'].fillna(0))
        
    if 'cos_angle' in features and 'cos_angle' not in input_df.columns and 'turning_angle' in input_df.columns:
        input_df['cos_angle'] = np.cos(input_df['turning_angle'].fillna(0))
        input_df['sin_angle'] = np.sin(input_df['turning_angle'].fillna(0))
        
    if 'is_nighttime' in features and 'is_nighttime' not in input_df.columns and 'time_of_day' in input_df.columns:
        input_df['is_nighttime'] = (input_df['time_of_day'] == 2).astype(int)
        
    if 'crop_attraction' in features and 'crop_attraction' not in input_df.columns and 'cropland_pct' in input_df.columns and 'ndvi' in input_df.columns:
        input_df['crop_attraction'] = input_df['cropland_pct'] * input_df['ndvi']
        
    if 'village_risk' in features and 'village_risk' not in input_df.columns and 'village_distance_m' in input_df.columns:
        input_df['village_risk'] = 1.0 / (input_df['village_distance_m'] + 1.0)
        
    for missing in [f for f in features if f not in input_df.columns]:
        input_df[missing] = 0.0
        
    # Keep only the needed features in exactly the right order
    input_df = input_df[features].fillna(0.0)
    
    # Pad to SEQ_LEN if we have fewer elements in the sequence
    if len(input_df) < SEQ_LEN:
        pad_len = SEQ_LEN - len(input_df)
        # Pad with zeros (neutral) instead of duplicating real data
        pad_df = pd.DataFrame(np.zeros((pad_len, len(input_df.columns))), columns=input_df.columns)
        input_df = pd.concat([pad_df, input_df], ignore_index=True)
    elif len(input_df) > SEQ_LEN:
        input_df = input_df.tail(SEQ_LEN)
        
    X_val = input_df.values
    
    # Scale exactly like training
    X_scaled = scaler.transform(X_val)
    
    # Convert to Tensor (shape: 1, seq_len, features)
    X_tensor = torch.tensor([X_scaled], dtype=torch.float32)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    X_tensor = X_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(X_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    
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
    
    # Store the context (features) used for this prediction
    inference_context = input_df.iloc[-1].to_dict()
    
    print(f"[{datetime.now().isoformat()}] Prediction complete.")
    return res_df, inference_context

if __name__ == "__main__":
    pass
