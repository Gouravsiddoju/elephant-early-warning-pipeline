import pandas as pd
import os

BASE_DIR = 'early_warning_pipeline'
FEATURE_CSV = os.path.join(BASE_DIR, 'feature_matrix.csv')
OUTPUT_CSV = os.path.join(BASE_DIR, 'grid_features.csv')

def extract_grid_features():
    print(f"Reading {FEATURE_CSV}...")
    # Load only necessary columns to save memory
    cols = ['from_grid', 'ndvi', 'cropland_pct', 'village_distance_m', 'rainfall_7d_mm']
    df = pd.read_csv(FEATURE_CSV, usecols=cols)
    
    print("Grouping by grid_id...")
    # Take the mean of environmental features per grid
    grid_features = df.groupby('from_grid').agg({
        'ndvi': 'mean',
        'cropland_pct': 'mean',
        'village_distance_m': 'first', # Usually static
        'rainfall_7d_mm': 'mean'
    }).reset_index()
    
    grid_features.rename(columns={'from_grid': 'grid_id'}, inplace=True)
    
    print(f"Saving to {OUTPUT_CSV}...")
    grid_features.to_csv(OUTPUT_CSV, index=False)
    print("Done!")

if __name__ == "__main__":
    extract_grid_features()
