import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import folium
from folium.plugins import HeatMap
from datetime import datetime
import json

def identify_at_risk_villages(predicted_grids_df: pd.DataFrame, osm_villages_gdf: gpd.GeoDataFrame, risk_radius_m=5000) -> pd.DataFrame:
    """
    For each predicted grid cell with prob > 0.3:
    Find all villages within risk_radius_m (5km).
    """
    print(f"[{datetime.now().isoformat()}] Identifying at-risk villages within {risk_radius_m}m...")
    
    high_risk_grids = predicted_grids_df[predicted_grids_df['probability'] > 0.3].copy()
    
    if high_risk_grids.empty or osm_villages_gdf.empty:
        return pd.DataFrame()
        
    # Drop rows with NaN centroids so points_from_xy doesn't crash
    valid_grids = high_risk_grids.dropna(subset=['centroid_lon', 'centroid_lat']).copy()
    if valid_grids.empty:
        return pd.DataFrame()

    grids_gdf = gpd.GeoDataFrame(
        valid_grids, 
        geometry=gpd.points_from_xy(valid_grids['centroid_lon'], valid_grids['centroid_lat']),
        crs="EPSG:4326"
    ).to_crs("EPSG:32734")
    
    if osm_villages_gdf.crs != "EPSG:32734":
        villages_utm = osm_villages_gdf.to_crs("EPSG:32734")
    else:
        villages_utm = osm_villages_gdf
        
    grids_gdf['geometry'] = grids_gdf.geometry.buffer(risk_radius_m)
    
    joined = gpd.sjoin(villages_utm, grids_gdf, how='inner', predicate='intersects')
    
    if joined.empty:
        return pd.DataFrame()
        
    name_col = 'name' if 'name' in joined.columns else 'fclass' if 'fclass' in joined.columns else 'id'
    
    results = []
    for _, row in joined.iterrows():
        v_geom = row.geometry
        
        g_lon = row['centroid_lon']
        g_lat = row['centroid_lat']
        g_pt = gpd.GeoSeries([Point(g_lon, g_lat)], crs="EPSG:4326").to_crs("EPSG:32734").iloc[0]
        
        dist_m = v_geom.distance(g_pt)
        prob = row['probability']
        
        risk_level = "LOW"
        if prob > 0.6:
            risk_level = "HIGH"
        elif prob >= 0.4:
            risk_level = "MEDIUM"
            
        results.append({
            'name': str(row.get(name_col, 'Unknown')),
            'distance_m': float(dist_m),
            'probability_score': float(prob),
            'risk_level': risk_level,
            'grid_id': row['grid_id'],
            'village_lon_utm': v_geom.centroid.x,
            'village_lat_utm': v_geom.centroid.y
        })
        
    results_df = pd.DataFrame(results)
    
    results_df = results_df.sort_values(by=['probability_score', 'distance_m'], ascending=[False, True]).drop_duplicates(subset=['name'])
    
    print(f"[{datetime.now().isoformat()}] Identified {len(results_df)} at-risk villages.")
    return results_df

def generate_alert_report(prediction_df: pd.DataFrame, villages_df: pd.DataFrame) -> dict:
    """
    Return structured alert payload.
    """
    print(f"[{datetime.now().isoformat()}] Generating JSON alert report...")
    
    if prediction_df.empty:
         return {}
         
    top_pred = prediction_df.iloc[0]
    
    villages_list = []
    if not villages_df.empty:
        for _, v in villages_df.iterrows():
            villages_list.append({
                'name': v['name'],
                'distance_m': round(v['distance_m'], 1),
                'risk_level': v['risk_level']
            })
            
    if len(villages_list) > 0:
        if any(v['risk_level'] == 'HIGH' for v in villages_list):
            action = "IMMEDIATE DISPATCH: Deploy ranger unit to intercept. Trigger local SMS warnings."
        elif any(v['risk_level'] == 'MEDIUM' for v in villages_list):
            action = "STANDBY: Alert village head. Monitor herd movement."
        else:
            action = "LOGGED: Low probability. Continue standard tracking."
    else:
        action = "SAFE: No settlements within 5km predicted path."

    alert = {
        'timestamp': datetime.now().isoformat(),
        'elephant_id': "DEMO_ID",  
        'current_grid': "UNKNOWN", 
        'prediction_horizon_hours': 48,
        'top_prediction': {
            'grid_id': str(top_pred['grid_id']),
            'probability': float(top_pred['probability']),
            'coordinates': [float(top_pred['centroid_lon']), float(top_pred['centroid_lat'])]
        },
        'at_risk_villages': villages_list,
        'recommended_action': action
    }
    
    with open('alert_output.json', 'w') as f:
        json.dump(alert, f, indent=4)
        
    print(f"[{datetime.now().isoformat()}] Alert report saved to alert_output.json")
    return alert

def plot_prediction_map(current_grid, predicted_grids, villages_df, grid_gdf=None):
    """
    Use folium to create interactive HTML map.
    """
    print(f"[{datetime.now().isoformat()}] Generating Folium prediction map...")
    
    center_lat = predicted_grids['centroid_lat'].mean()
    center_lon = predicted_grids['centroid_lon'].mean()
    
    if np.isnan(center_lat) or np.isnan(center_lon):
        center_lat, center_lon = -19.5, 23.5
        
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="OpenStreetMap")
    
    heat_data = []
    for _, row in predicted_grids.iterrows():
        if not np.isnan(row['centroid_lat']) and not np.isnan(row['centroid_lon']):
            heat_data.append([row['centroid_lat'], row['centroid_lon'], row['probability']])
            
            folium.CircleMarker(
                location=[row['centroid_lat'], row['centroid_lon']],
                radius=8,
                color="orange",
                fill=True,
                fill_color="orange",
                fill_opacity=min(1.0, row['probability'] * 2),
                popup=f"Grid: {row['grid_id']}<br>Prob: {row['probability']:.2f}"
            ).add_to(m)
            
    if heat_data:
        HeatMap(heat_data, min_opacity=0.3, radius=25, blur=15, max_zoom=1).add_to(m)
        
    if not villages_df.empty and 'village_lon_utm' in villages_df.columns:
        pts = gpd.GeoSeries([Point(x, y) for x, y in zip(villages_df['village_lon_utm'], villages_df['village_lat_utm'])], crs="EPSG:32734")
        pts_wgs84 = pts.to_crs("EPSG:4326")
        
        for idx, (_, row) in enumerate(villages_df.iterrows()):
            v_lon = pts_wgs84.iloc[idx].x
            v_lat = pts_wgs84.iloc[idx].y
            
            color = "red" if row['risk_level'] == 'HIGH' else "orange" if row['risk_level'] == 'MEDIUM' else "green"
            
            folium.Marker(
                location=[v_lat, v_lon],
                icon=folium.Icon(color=color, icon="info-sign"),
                popup=f"<b>Village:</b> {row['name']}<br><b>Risk:</b> {row['risk_level']}<br><b>Distance:</b> {row['distance_m']:.0f}m"
            ).add_to(m)
            
    m.save('alert_map.html')
    print(f"[{datetime.now().isoformat()}] Map saved to alert_map.html")

if __name__ == "__main__":
    pass
