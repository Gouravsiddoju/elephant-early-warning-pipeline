import pandas as pd
import geopandas as gpd
from datetime import datetime

def load_gps_data(filepath: str) -> pd.DataFrame:
    """
    Load Dryad CSV (semicolon-separated).
    Parse Date_Time as datetime with timezone UTC.
    Sort by [id, Date_Time] and validate required columns.
    """
    print(f"[{datetime.now().isoformat()}] Loading GPS data from {filepath}...")
    
    # Load Dryad CSV — auto-detect separator (real file is comma-separated)
    df = pd.read_csv(filepath, sep=None, engine='python')
    
    # Parse Date_Time — real data uses DD-MM-YYYY HH:MM format
    # Try multiple formats gracefully
    for fmt in ['%d-%m-%Y %H:%M', '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S']:
        try:
            df['Date_Time'] = pd.to_datetime(df['Date_Time'], format=fmt, utc=True)
            break
        except (ValueError, TypeError):
            continue
    else:
        # Fallback: let pandas infer
        df['Date_Time'] = pd.to_datetime(df['Date_Time'], utc=True, dayfirst=True)
    
    # Sort by [id, Date_Time]
    df = df.sort_values(by=['id', 'Date_Time']).reset_index(drop=True)
    
    # Validate: no nulls in Date_Time, Dist, Angle, id
    required_cols = ['Date_Time', 'Dist', 'Angle', 'id']
    for col in required_cols:
         if col not in df.columns:
              raise ValueError(f"Missing required column: {col}")
         
    initial_len = len(df)
    df = df.dropna(subset=required_cols)
    if len(df) < initial_len:
         print(f"[{datetime.now().isoformat()}] Dropped {initial_len - len(df)} rows with missing required values.")
         
    # Print summary: n_individuals, date_range, n_fixes_per_elephant
    n_individuals = df['id'].nunique()
    date_min = df['Date_Time'].min()
    date_max = df['Date_Time'].max()
    fixes_per_elephant = df.groupby('id').size().to_dict()
    
    print(f"[{datetime.now().isoformat()}] GPS Data Summary:")
    print(f"  - Individuals: {n_individuals}")
    print(f"  - Date Range: {date_min} to {date_max}")
    print(f"  - Fixes per elephant:")
    for eid, count in fixes_per_elephant.items():
        print(f"      {eid}: {count} fixes")
    
    return df

def load_shapefile(filepath: str) -> gpd.GeoDataFrame:
    """
    Load Botswana shapefile.
    Reproject to EPSG:32734 (UTM Zone 34S — correct for Botswana).
    """
    print(f"[{datetime.now().isoformat()}] Loading shapefile from {filepath}...")
    gdf = gpd.read_file(filepath)
    
    # Reproject to EPSG:32734
    if gdf.crs != "EPSG:32734":
        print(f"[{datetime.now().isoformat()}] Reprojecting from {gdf.crs} to EPSG:32734...")
        gdf = gdf.to_crs("EPSG:32734")
    else:
        print(f"[{datetime.now().isoformat()}] Shapefile already in EPSG:32734.")
        
    return gdf

def validate_spatial_bounds(gdf: pd.DataFrame, lon_min=23, lon_max=28, lat_min=-22, lat_max=-17) -> pd.DataFrame:
    """
    Assert all GPS points fall within Botswana bounding box.
    Flag and remove outliers, log count removed.
    """
    print(f"[{datetime.now().isoformat()}] Validating spatial bounds within lon:[{lon_min}, {lon_max}] lat:[{lat_min}, {lat_max}]...")
    
    initial_len = len(gdf)
    
    # Typical names for longitude and latitude
    lon_col = None
    lat_col = None
    for col in gdf.columns:
        col_lower = col.lower()
        if col_lower in ['lon', 'long', 'longitude', 'location-long']:
            lon_col = col
        if col_lower in ['lat', 'latitude', 'location-lat']:
            lat_col = col
            
    if lon_col is None or lat_col is None:
        print(f"[{datetime.now().isoformat()}] WARNING: Could not auto-detect longitude/latitude columns for spatial validation. Returning unmodified DataFrame.")
        return gdf
        
    bounds_mask = (
        (gdf[lon_col] >= lon_min) & (gdf[lon_col] <= lon_max) &
        (gdf[lat_col] >= lat_min) & (gdf[lat_col] <= lat_max)
    )
    
    outliers = ~bounds_mask
    outlier_count = outliers.sum()
    
    if outlier_count > 0:
        print(f"[{datetime.now().isoformat()}] Flagged and removing {outlier_count} spatial outliers outside bounds.")
        gdf = gdf[bounds_mask].copy()
    else:
        print(f"[{datetime.now().isoformat()}] All points within valid bounds.")
        
    return gdf

if __name__ == "__main__":
    print("Testing data_loader.py functions. (Please provide actual paths to test properly in main.py)")
