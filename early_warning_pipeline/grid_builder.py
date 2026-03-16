import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon
from datetime import datetime

def build_grid(bounds, cell_size_m=500) -> gpd.GeoDataFrame:
    """
    Create a regular 500m x 500m grid over Botswana study area.
    Each cell gets a unique grid_id (e.g., "R012_C045").
    CRS: EPSG:32734
    
    bounds should be a tuple of (minx, miny, maxx, maxy) in EPSG:32734
    """
    print(f"[{datetime.now().isoformat()}] Building {cell_size_m}m x {cell_size_m}m spatial grid...")
    minx, miny, maxx, maxy = bounds
    
    # Calculate number of cells
    cols = int(np.ceil((maxx - minx) / cell_size_m))
    rows = int(np.ceil((maxy - miny) / cell_size_m))
    
    print(f"[{datetime.now().isoformat()}] Grid dimensions: {rows} rows x {cols} cols (Total: {rows * cols} cells)")
    
    polygons = []
    grid_ids = []
    
    # Generate cells
    for r in range(rows):
        y_top = maxy - (r * cell_size_m)
        y_bottom = y_top - cell_size_m
        for c in range(cols):
            x_left = minx + (c * cell_size_m)
            x_right = x_left + cell_size_m
            
            polygons.append(Polygon([(x_left, y_top), (x_right, y_top), (x_right, y_bottom), (x_left, y_bottom)]))
            grid_ids.append(f"R{r:04d}_C{c:04d}")
            
    grid_gdf = gpd.GeoDataFrame({'grid_id': grid_ids}, geometry=polygons, crs="EPSG:32734")
    
    # Pre-compute and save centroids in WSG84 for the predictor
    centroids_wgs84 = grid_gdf.geometry.centroid.to_crs("EPSG:4326")
    pd.DataFrame({
        'grid_id': grid_ids,
        'centroid_lon': centroids_wgs84.x,
        'centroid_lat': centroids_wgs84.y
    }).to_csv('grid_centroids.csv', index=False)
    
    print(f"[{datetime.now().isoformat()}] Grid built successfully.")
    
    return grid_gdf

def assign_gps_to_grid(gps_df: pd.DataFrame, grid_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Spatial join each GPS fix to its grid cell.
    Add grid_id, grid_centroid_lon, grid_centroid_lat to gps_df.
    """
    print(f"[{datetime.now().isoformat()}] Assigning GPS points to grid cells...")
    
    lon_col = next((c for c in gps_df.columns if c.lower() in ['lon', 'long', 'longitude', 'location-long']), None)
    lat_col = next((c for c in gps_df.columns if c.lower() in ['lat', 'latitude', 'location-lat']), None)
    
    if not lon_col or not lat_col:
         raise ValueError("Could not find longitude or latitude columns in gps_df")
         
    # Create GeoDataFrame for GPS points (assuming WGS84 EPSG:4326 initially)
    gps_gdf = gpd.GeoDataFrame(
        gps_df, 
        geometry=gpd.points_from_xy(gps_df[lon_col], gps_df[lat_col]),
        crs="EPSG:4326"
    )
    
    # Reproject to match grid (EPSG:32734)
    gps_gdf = gps_gdf.to_crs("EPSG:32734")
    
    # Spatial Join
    joined = gpd.sjoin(gps_gdf, grid_gdf, how="left", predicate="intersects")
    
    # Compute centroids in UTM, then convert just those to WGS84 for the lon/lat cols
    grid_centroids = grid_gdf.copy()
    centroids_utm = grid_centroids.geometry.centroid
    centroids_wgs84 = centroids_utm.to_crs("EPSG:4326")
    
    grid_centroids['grid_centroid_lon'] = centroids_wgs84.x
    grid_centroids['grid_centroid_lat'] = centroids_wgs84.y
    
    centroid_map = grid_centroids.set_index('grid_id')[['grid_centroid_lon', 'grid_centroid_lat']]
    
    # Merge centroids onto joined DataFrame
    joined = joined.merge(centroid_map, on='grid_id', how='left')
    
    # After join, explicitly catch and log failures
    missing_mask = joined['grid_id'].isna()
    missing_count = missing_mask.sum()
    if missing_count > 0:
        print(f"[{datetime.now().isoformat()}] WARNING: {missing_count} GPS points did not fall into any grid cell.")
        print(f"  Sample out-of-bounds points:")
        print(joined[missing_mask][[lon_col, lat_col]].head(5).to_string())
        print(f"  -> Expand STUDY_LON/LAT bounds in main.py to fix this.")
    
    # Drop unassigned points — never let NaN propagate
    joined = joined.dropna(subset=['grid_id'])
    
    # Convert back to standard DataFrame
    final_df = pd.DataFrame(joined.drop(columns=['geometry']))
    
    return final_df

def compute_transitions(gps_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each elephant, for each consecutive pair of fixes:
    - from_grid, to_grid, time_delta_hours, step_dist_m, turning_angle
    """
    print(f"[{datetime.now().isoformat()}] Computing grid transitions...")
    
    # Ensure sorted
    gps_df = gps_df.sort_values(['id', 'Date_Time']).copy()
    
    # Create shifted columns
    gps_df['next_grid'] = gps_df.groupby('id')['grid_id'].shift(-1)
    gps_df['next_time'] = gps_df.groupby('id')['Date_Time'].shift(-1)
    
    # Calculate time delta in hours
    gps_df['time_delta_hours'] = (gps_df['next_time'] - gps_df['Date_Time']).dt.total_seconds() / 3600.0
    
    # Filter out last row per elephant (where next is NaT)
    transitions = gps_df.dropna(subset=['next_grid', 'next_time']).copy()
    
    # Rename for clarity
    transitions = transitions.rename(columns={
        'id': 'elephant_id',
        'grid_id': 'from_grid',
        'next_grid': 'to_grid'
    })
    
    # Some datasets capitalize 'Dist' and 'Angle'. Make sure we map to 'step_dist_m' and 'turning_angle'
    if 'Dist' in transitions.columns:
        transitions['step_dist_m'] = transitions['Dist']
    
    if 'Angle' in transitions.columns:
        transitions['turning_angle'] = transitions['Angle']
        
    keep_cols = [
        'elephant_id', 'Date_Time', 'from_grid', 'to_grid', 
        'time_delta_hours', 'step_dist_m', 'turning_angle',
        'grid_centroid_lon', 'grid_centroid_lat'
    ]
    
    # Add any extra base columns that are originally there
    for extra in ['Season', 'TimeofDay']:
        if extra in transitions.columns:
            keep_cols.append(extra)
            
    # filter only existing cols to avoid KeyError
    keep_cols = [c for c in keep_cols if c in transitions.columns]
            
    transitions_final = transitions[keep_cols].copy()
    print(f"[{datetime.now().isoformat()}] Computed {len(transitions_final)} transitions.")
    
    return transitions_final

if __name__ == "__main__":
    pass
