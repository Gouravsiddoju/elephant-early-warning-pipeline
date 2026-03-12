import pandas as pd
import geopandas as gpd
from datetime import datetime

def extract_osm_features(osm_shapefile_path: str, grid_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Load OSM Botswana shapefile.
    Extract: settlement points, road lines, cropland polygons.
    Compute village_distance_m, road_density, cropland_pct, has_powerline for each grid cell.
    """
    print(f"[{datetime.now().isoformat()}] Loading OSM features from {osm_shapefile_path}...")
    
    try:
        osm_gdf = gpd.read_file(osm_shapefile_path)
    except Exception as e:
        print(f"[{datetime.now().isoformat()}] WARNING: Could not load OSM shapefile. Returning grid with default zeroes for OSM features. Error: {e}")
        # Insert dummy columns
        grid_gdf['village_distance_m'] = 10000.0  # Safe default long distance
        grid_gdf['road_density'] = 0.0
        grid_gdf['cropland_pct'] = 0.0
        grid_gdf['has_powerline'] = 0
        return grid_gdf
        
    # Reproject to UTM 34S to match grid
    if osm_gdf.crs != "EPSG:32734":
        osm_gdf = osm_gdf.to_crs("EPSG:32734")
        
    # Separate layers based on standard OSM tags 
    if 'fclass' in osm_gdf.columns:
        villages = osm_gdf[osm_gdf['fclass'].isin(['village', 'town', 'city', 'hamlet'])]
        roads = osm_gdf[osm_gdf['fclass'].isin(['primary', 'secondary', 'tertiary', 'trunk', 'motorway', 'track', 'path'])]
    elif 'highway' in osm_gdf.columns:
        villages = osm_gdf[osm_gdf['place'].notna()]
        roads = osm_gdf[osm_gdf['highway'].notna()]
    else:
        villages = osm_gdf[osm_gdf.geometry.type == 'Point']
        roads = osm_gdf[osm_gdf.geometry.type.isin(['LineString', 'MultiLineString'])]
        
    if 'landuse' in osm_gdf.columns:
        cropland = osm_gdf[osm_gdf['landuse'].isin(['farmland', 'orchard', 'vineyard'])]
    elif 'fclass' in osm_gdf.columns:
        cropland = osm_gdf[osm_gdf['fclass'].isin(['farmland', 'farm'])]
    else:
        cropland = osm_gdf[osm_gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])]
        
    powerlines = osm_gdf[osm_gdf['power'] == 'line'] if 'power' in osm_gdf.columns else gpd.GeoDataFrame(geometry=[])

    print(f"[{datetime.now().isoformat()}] Computing spatial features for grid cells (this may take time)...")
    
    # 1. village_distance_m
    if not villages.empty:
        temp_grid = grid_gdf.copy()
        temp_grid.geometry = temp_grid.geometry.centroid
        
        nearest_villages = gpd.sjoin_nearest(temp_grid, villages, how='left', distance_col='village_distance_m')
        nearest_villages = nearest_villages.drop_duplicates(subset=['grid_id'])
        
        grid_gdf = grid_gdf.merge(nearest_villages[['grid_id', 'village_distance_m']], on='grid_id', how='left')
    else:
        grid_gdf['village_distance_m'] = 10000.0
        
    grid_gdf['village_distance_m'] = grid_gdf['village_distance_m'].fillna(10000.0)
    
    # 2. road_density and others
    grid_area_km2 = (500 * 500) / 1e6
    grid_gdf['road_density'] = 0.0
    grid_gdf['cropland_pct'] = 0.0
    grid_gdf['has_powerline'] = 0
    
    if not roads.empty:
        print("Calculating road density...")
        intersections = gpd.overlay(grid_gdf[['grid_id', 'geometry']], roads[['geometry']], how='intersection')
        intersections['road_len_km'] = intersections.geometry.length / 1000.0
        road_sums = intersections.groupby('grid_id')['road_len_km'].sum().reset_index()
        road_sums['road_density'] = road_sums['road_len_km'] / grid_area_km2
        grid_gdf = grid_gdf.drop(columns=['road_density'], errors='ignore').merge(road_sums[['grid_id', 'road_density']], on='grid_id', how='left')
        grid_gdf['road_density'] = grid_gdf['road_density'].fillna(0.0)

    if not cropland.empty:
        print("Calculating cropland percentage...")
        intersections = gpd.overlay(grid_gdf[['grid_id', 'geometry']], cropland[['geometry']], how='intersection')
        intersections['crop_area_sqm'] = intersections.geometry.area
        crop_sums = intersections.groupby('grid_id')['crop_area_sqm'].sum().reset_index()
        crop_sums['cropland_pct'] = crop_sums['crop_area_sqm'] / (500 * 500)
        grid_gdf = grid_gdf.drop(columns=['cropland_pct'], errors='ignore').merge(crop_sums[['grid_id', 'cropland_pct']], on='grid_id', how='left')
        grid_gdf['cropland_pct'] = grid_gdf['cropland_pct'].fillna(0.0)
        
    if not powerlines.empty:
        print("Checking powerlines...")
        joined = gpd.sjoin(grid_gdf[['grid_id', 'geometry']], powerlines, how='inner', predicate='intersects')
        has_power_grids = joined['grid_id'].unique()
        grid_gdf['has_powerline'] = grid_gdf['grid_id'].isin(has_power_grids).astype(int)
        
    print(f"[{datetime.now().isoformat()}] OSM extraction complete.")
    return grid_gdf

def merge_human_features(gps_df: pd.DataFrame, grid_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Join human features onto GPS dataframe via grid_id.
    """
    print(f"[{datetime.now().isoformat()}] Merging human features to gps data...")
    features_cols = ['grid_id', 'village_distance_m', 'road_density', 'cropland_pct', 'has_powerline']
    
    missing = [c for c in features_cols if c not in grid_gdf.columns]
    if missing:
        features_cols = [c for c in features_cols if c in grid_gdf.columns]
        
    merged_df = gps_df.merge(grid_gdf[features_cols], on='grid_id', how='left')
    
    return merged_df

if __name__ == "__main__":
    pass
