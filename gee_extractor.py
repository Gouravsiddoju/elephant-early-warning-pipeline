import pandas as pd
import numpy as np
import ee
from datetime import datetime, timedelta
from tqdm import tqdm
import os
import joblib

# Defer EE initialization to avoid import-time crash when not authenticated
_EE_INITIALIZED = False

def _ensure_ee():
    """Lazy-initialize Earth Engine on first use."""
    global _EE_INITIALIZED
    if not _EE_INITIALIZED:
        try:
            ee.Initialize()
            _EE_INITIALIZED = True
        except Exception as e:
            print(f"[WARNING] Earth Engine not authenticated: {e}")
            print("[WARNING] GEE features will be skipped. Run 'earthengine authenticate' to enable them.")

def extract_ndvi(date, lon, lat):
    """
    Query MODIS/061/MOD13Q1 for the 16-day window containing date.
    Sample NDVI at (lon, lat), scale by 0.0001. Returns -9999 if missing.
    """
    _ensure_ee()
    if not _EE_INITIALIZED:
        return -9999.0
    try:
        if isinstance(date, str):
            date = pd.to_datetime(date)
            
        start_date = (date - timedelta(days=8)).strftime('%Y-%m-%d')
        end_date = (date + timedelta(days=8)).strftime('%Y-%m-%d')
        
        point = ee.Geometry.Point([lon, lat])
        collection = ee.ImageCollection('MODIS/061/MOD13Q1') \
            .filterBounds(point) \
            .filterDate(start_date, end_date) \
            .select('NDVI')
        image = collection.median()
        value = image.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=point,
            scale=250
        ).get('NDVI').getInfo()
        if value is not None:
            return value * 0.0001
        return -9999.0
    except Exception:
        return -9999.0

def extract_rainfall(date, lon, lat, window_days=7):
    """
    Query UCSB-CHG/CHIRPS/DAILY for 7-day window before date.
    Returns total mm as float.
    """
    _ensure_ee()
    if not _EE_INITIALIZED:
        return 0.0
    try:
        if isinstance(date, str):
            date = pd.to_datetime(date)
        start_date = (date - timedelta(days=window_days)).strftime('%Y-%m-%d')
        end_date = date.strftime('%Y-%m-%d')
        point = ee.Geometry.Point([lon, lat])
        collection = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY') \
            .filterBounds(point) \
            .filterDate(start_date, end_date) \
            .select('precipitation')
        image = collection.sum()
        value = image.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=point,
            scale=5566
        ).get('precipitation').getInfo()
        if value is not None:
            return float(value)
        return 0.0
    except Exception:
        return 0.0

def extract_landcover(lon, lat, year):
    """
    Query MODIS/061/MCD12Q1 for given year.
    Returns LC_Type1 class integer at (lon, lat).
    """
    _ensure_ee()
    if not _EE_INITIALIZED:
        return -9999
    try:
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        point = ee.Geometry.Point([lon, lat])
        collection = ee.ImageCollection('MODIS/061/MCD12Q1') \
            .filterBounds(point) \
            .filterDate(start_date, end_date) \
            .select('LC_Type1')
        image = collection.first()
        value = image.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=point,
            scale=500
        ).get('LC_Type1').getInfo()
        if value is not None:
            return int(value)
        return -9999
    except Exception:
        return -9999

def batch_extract_features(gps_df: pd.DataFrame, cache_file='gee_cache.pkl') -> pd.DataFrame:
    """
    For every unique (date, grid_centroid_lon, grid_centroid_lat):
    - Extract NDVI, rainfall, landcover
    Cache results to avoid repeated GEE calls.
    Merge back onto gps_df.
    """
    _ensure_ee()
    if not _EE_INITIALIZED:
        print(f"[{datetime.now().isoformat()}] GEE not authenticated — skipping satellite features. Run 'earthengine authenticate' to enable.")
        gps_df['ndvi'] = 0.5
        gps_df['rainfall_7d_mm'] = 0.0
        gps_df['landcover_class'] = -9999
        gps_df['is_forest'] = 0
        gps_df['is_cropland'] = 0
        return gps_df
    print(f"[{datetime.now().isoformat()}] Starting batch GEE extraction...")
    
    # Load cache if it exists
    cache = {}
    if os.path.exists(cache_file):
        print(f"[{datetime.now().isoformat()}] Loading GEE cache from {cache_file}...")
        try:
            cache = joblib.load(cache_file)
        except:
            print("Failed to load cache. Starting fresh.")
            
    # Need date part for querying 
    gps_df['query_date'] = gps_df['Date_Time'].dt.date
    gps_df['query_year'] = gps_df['Date_Time'].dt.year
    
    # Get unique combinations
    unique_queries = gps_df[['query_date', 'query_year', 'grid_centroid_lon', 'grid_centroid_lat']].drop_duplicates()
    total_queries = len(unique_queries)
    
    print(f"[{datetime.now().isoformat()}] Found {total_queries} unique spatial-temporal query points.")
    
    results = []
    
    for _, row in tqdm(unique_queries.iterrows(), total=total_queries, desc="GEE API Calls"):
        date_str = str(row['query_date'])
        year = row['query_year']
        lon = row['grid_centroid_lon']
        lat = row['grid_centroid_lat']
        
        cache_key = f"{date_str}_{lon:.5f}_{lat:.5f}"
        
        if cache_key in cache:
            results.append(cache[cache_key])
            continue
            
        # Needs API call
        ndvi = extract_ndvi(date_str, lon, lat)
        rain = extract_rainfall(date_str, lon, lat)
        lc = extract_landcover(lon, lat, year)
        
        res = {
            'query_date': row['query_date'],
            'grid_centroid_lon': lon,
            'grid_centroid_lat': lat,
            'ndvi': ndvi,
            'rainfall_7d_mm': rain,
            'landcover_class': lc
        }
        
        cache[cache_key] = res
        results.append(res)
        
    # Save updated cache periodically/at end
    print(f"[{datetime.now().isoformat()}] Saving GEE cache...")
    joblib.dump(cache, cache_file)
    
    results_df = pd.DataFrame(results)
    
    # Impute missing NDVI with spatial median (here we use simple median as proxy)
    if (results_df['ndvi'] == -9999.0).any():
        med = results_df[results_df['ndvi'] != -9999.0]['ndvi'].median()
        results_df.loc[results_df['ndvi'] == -9999.0, 'ndvi'] = med
        
    # Landcover flags
    results_df['is_forest'] = results_df['landcover_class'].isin([1, 2, 3, 4, 5]).astype(int)
    results_df['is_cropland'] = (results_df['landcover_class'] == 12).astype(int)
    
    print(f"[{datetime.now().isoformat()}] Merging GEE features back to main dataset...")
    # Merge back onto gps_df
    final_df = gps_df.merge(
        results_df, 
        on=['query_date', 'grid_centroid_lon', 'grid_centroid_lat'], 
        how='left'
    )
    
    # Drop temp cols
    final_df = final_df.drop(columns=['query_date', 'query_year'])
    
    return final_df

if __name__ == "__main__":
    pass
