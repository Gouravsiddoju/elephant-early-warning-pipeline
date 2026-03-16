import pandas as pd
import numpy as np
import geopandas as gpd
from datetime import datetime

def build_feature_matrix(gps_df: pd.DataFrame, transitions_df: pd.DataFrame, grid_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Merge all 5 dataset layers onto transitions_df.
    DATASET A: transitions_df
    DATASET B/D: from gps_df (GEE + Dryad)
    DATASET C: from grid_gdf
    DATASET E: from memory
    """
    # Only pull columns NOT already in transitions_df
    existing_cols = df.columns.tolist()
    cols_to_pull = [
        'elephant_id', 'Date_Time', 
        'ndvi', 'landcover_class', 'is_forest', 'is_cropland', 'rainfall_7d_mm',
        'Season', 'TimeofDay',
        'visit_count', 'last_visit_days_ago', 'is_home_range_core'
    ]
    
    # Filter to what exists in gps_df AND isn't already in df (except join keys)
    cols_to_pull = [c for c in cols_to_pull if c in gps_rename.columns and (c in ['elephant_id', 'Date_Time'] or c not in existing_cols)]
    
    print(f"[{datetime.now().isoformat()}] Merging {len(cols_to_pull)-2} external features...")
    df = df.merge(gps_rename[cols_to_pull], on=['elephant_id', 'Date_Time'], how='left')
    
    df['month'] = df['Date_Time'].dt.month
    df['hour'] = df['Date_Time'].dt.hour
    
    # Standardize names to lowercase
    rename_map = {
        'Season': 'season',
        'TimeofDay': 'time_of_day',
        'Dist': 'step_dist_m',
        'Angle': 'turning_angle'
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    
    # Check for any remaining _x/_y suffixes and clean them
    for col in df.columns:
        if col.endswith('_x'):
            clean_name = col[:-2]
            if f"{clean_name}_y" in df.columns:
                df = df.drop(columns=[f"{clean_name}_y"])
                df = df.rename(columns={col: clean_name})
        
    grid_cols = ['grid_id', 'village_distance_m', 'road_density', 'cropland_pct']
    grid_cols = [c for c in grid_cols if c in grid_gdf.columns]
    
    if grid_cols:
        grid_features = grid_gdf[grid_cols].copy()
        df = df.merge(grid_features, left_on='from_grid', right_on='grid_id', how='left')
        df = df.drop(columns=['grid_id'])

    print(f"[{datetime.now().isoformat()}] Final matrix shape: {df.shape[0]} rows, {df.shape[1]} features")
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features:
    - log_step_dist = log1p(step_dist_m)
    - cos_angle = cos(turning_angle)
    - sin_angle = sin(turning_angle)
    - is_nighttime = (time_of_day == 2).astype(int)
    - crop_attraction = cropland_pct * ndvi (interaction term)
    - village_risk = 1 / (village_distance_m + 1) (proximity score)
    """
    print(f"[{datetime.now().isoformat()}] Engineering derived features...")
    
    if 'step_dist_m' in df.columns:
        df['log_step_dist'] = np.log1p(df['step_dist_m'].fillna(0))
        
    if 'turning_angle' in df.columns:
        df['cos_angle'] = np.cos(df['turning_angle'].fillna(0))
        df['sin_angle'] = np.sin(df['turning_angle'].fillna(0))
        
    if 'time_of_day' in df.columns:
        df['is_nighttime'] = (df['time_of_day'] == 2).astype(int)
        
    if 'cropland_pct' in df.columns and 'ndvi' in df.columns:
        df['crop_attraction'] = df['cropland_pct'] * df['ndvi']
        
    if 'village_distance_m' in df.columns:
        df['village_risk'] = 1.0 / (df['village_distance_m'] + 1.0)
        
    # Drop rows with NaN in target
    initial_len = len(df)
    df = df.dropna(subset=['to_grid'])
    if len(df) < initial_len:
        print(f"[{datetime.now().isoformat()}] Dropped {initial_len - len(df)} rows missing target 'to_grid'.")
        
    # Preserve grid ID columns before bulk fillna
    grid_cols = ['from_grid', 'to_grid', 'elephant_id', 'Date_Time']
    grid_cols = [c for c in grid_cols if c in df.columns]
    saved = df[grid_cols].copy()
    
    # Only fill NaN in numeric feature columns
    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # Restore grid cols intact
    df[grid_cols] = saved
        
    print(f"[{datetime.now().isoformat()}] Engineered feature matrix completed.")
    return df

if __name__ == "__main__":
    pass
