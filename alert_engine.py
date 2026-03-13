import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import folium
from folium.plugins import HeatMap
from datetime import datetime
import json
import numpy as np

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

def plot_prediction_map(current_grid, predicted_grids, villages_df, grid_gdf=None, historical_path=None):
    """
    Premium interactive prediction map:
      - Predicted grid cells as WGS84 polygons (colour-coded by rank, always visible)
      - Full background grid overlay  (default OFF, user can toggle ON)
      - Trajectory lines from current position → each prediction
      - 5 km risk-radius dashed circles
      - Rank badges (#1-#5) on each cell centroid
      - Village markers with risk colours
      - Historical Path (if provided) drawn as a solid line
      - Detailed dark info-panel + probability bar chart
      - Layer control (basemaps + all feature groups)
    """
    print(f"[{datetime.now().isoformat()}] Generating Folium prediction map...")

    RANK_COLORS  = ["#FF3B30", "#FF9500", "#FFCC00", "#34C759", "#007AFF"]
    RANK_OPACITY = [0.80,       0.70,      0.60,      0.50,      0.40     ]

    # ── Pre-compute WGS84 grid lookup ─────────────────────────────────────────
    # Convert the entire grid to WGS84 once (avoids per-cell reprojection)
    grid_wgs84 = None
    if grid_gdf is not None:
        print(f"[{datetime.now().isoformat()}] Reprojecting grid to WGS84...")
        grid_wgs84 = grid_gdf.to_crs("EPSG:4326").set_index('grid_id')
        print(f"[{datetime.now().isoformat()}] Grid WGS84 reprojection complete ({len(grid_wgs84)} cells).")

    # ── Map centre ────────────────────────────────────────────────────────────
    valid = predicted_grids.dropna(subset=['centroid_lat', 'centroid_lon'])
    center_lat = valid['centroid_lat'].mean() if not valid.empty else -18.5
    center_lon = valid['centroid_lon'].mean() if not valid.empty else 24.5

    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles=None)

    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr='&copy; <a href="https://carto.com/">CARTO</a>',
        name="🌑 Dark (CARTO)", max_zoom=19
    ).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="🛰️ Satellite (ESRI)", max_zoom=19
    ).add_to(m)
    folium.TileLayer(tiles="OpenStreetMap", name="🗺️ Street Map (OSM)").add_to(m)

    # ── Feature groups ────────────────────────────────────────────────────────
    fg_grid_overlay = folium.FeatureGroup(name="🔲 Full Grid Overlay (5 km cells)", show=True)
    fg_grids        = folium.FeatureGroup(name="🎯 Predicted Grid Cells",           show=True)
    fg_traj         = folium.FeatureGroup(name="➡️ Trajectory Lines",               show=True)
    fg_radius       = folium.FeatureGroup(name="🔴 5 km Risk Radius",               show=True)
    fg_history      = folium.FeatureGroup(name="🕰️ Historical Path",                show=True)
    fg_villages     = folium.FeatureGroup(name="🏘️ Villages",                       show=True)
    fg_labels       = folium.FeatureGroup(name="🏷️ Rank Labels",                    show=True)
    fg_current      = folium.FeatureGroup(name="🐘 Current Position",               show=True)

    # ── Current elephant position ─────────────────────────────────────────────
    cur_lat, cur_lon = center_lat - 0.3, center_lon - 0.3
    if grid_wgs84 is not None and current_grid in grid_wgs84.index:
        centroid = grid_wgs84.loc[current_grid].geometry.centroid
        cur_lon, cur_lat = centroid.x, centroid.y

    elephant_html = (
        "<div style='background:#FF3B30;border:3px solid white;border-radius:50%;"
        "width:40px;height:40px;display:flex;align-items:center;justify-content:center;"
        "font-size:22px;box-shadow:0 0 16px rgba(255,59,48,0.9);'>🐘</div>"
    )
    folium.Marker(
        location=[cur_lat, cur_lon],
        icon=folium.DivIcon(html=elephant_html, icon_size=(40, 40), icon_anchor=(20, 20)),
        tooltip=f"<b>🐘 Current Position</b><br>Grid: <code>{current_grid}</code><br>Lat: {cur_lat:.5f}°, Lon: {cur_lon:.5f}°",
        popup=(
            f"<div style='font-family:Arial;min-width:200px'>"
            f"<b style='font-size:14px'>🐘 Current Elephant Position</b><hr style='margin:4px 0'>"
            f"<b>Grid ID:</b> {current_grid}<br>"
            f"<b>Latitude:</b> {cur_lat:.5f}°<br>"
            f"<b>Longitude:</b> {cur_lon:.5f}°<br>"
            f"<b>Prediction Horizon:</b> 48 hours"
            f"</div>"
        )
    ).add_to(fg_current)

    # ── Full background grid overlay (GeoJSON, off by default) ───────────────
    if grid_wgs84 is not None:
        # Limit to a bounding-box around the prediction area to keep file size sane
        pred_lats = valid['centroid_lat'].tolist() + [cur_lat]
        pred_lons = valid['centroid_lon'].tolist() + [cur_lon]
        bb_lat_min = min(pred_lats) - 1.5
        bb_lat_max = max(pred_lats) + 1.5
        bb_lon_min = min(pred_lons) - 1.5
        bb_lon_max = max(pred_lons) + 1.5

        nearby_cells = grid_wgs84[
            grid_wgs84.geometry.centroid.y.between(bb_lat_min, bb_lat_max) &
            grid_wgs84.geometry.centroid.x.between(bb_lon_min, bb_lon_max)
        ].reset_index()

        print(f"[{datetime.now().isoformat()}] Adding {len(nearby_cells)} grid cells to background overlay...")

        if not nearby_cells.empty:
            folium.GeoJson(
                nearby_cells.__geo_interface__,
                name=None,
                style_function=lambda _: {
                    "color": "#4488FF",
                    "weight": 0.6,
                    "fillColor": "#88AAFF",
                    "fillOpacity": 0.04,
                    "opacity": 0.35
                },
                tooltip=folium.GeoJsonTooltip(fields=["grid_id"], aliases=["Grid Cell:"], sticky=False),
                highlight_function=lambda _: {"fillOpacity": 0.25, "weight": 1.5, "color": "#FFFFFF"}
            ).add_to(fg_grid_overlay)

    # ── Predicted grid cells + risk circles + rank labels ────────────────────
    rank_stats = {}

    for rank_0, (_, row) in enumerate(valid.iterrows()):
        rank    = int(row.get('rank', rank_0 + 1))
        clat    = float(row['centroid_lat'])
        clon    = float(row['centroid_lon'])
        prob    = float(row['probability'])
        grid_id = str(row['grid_id'])
        color   = RANK_COLORS[rank - 1]   if rank <= 5 else "#FFFFFF"
        opac    = RANK_OPACITY[rank - 1]  if rank <= 5 else 0.3

        rank_stats[rank] = {'grid_id': grid_id, 'prob': prob, 'lat': clat, 'lon': clon}

        # -- Polygon from WGS84 grid lookup (always visible) --
        if grid_wgs84 is not None and grid_id in grid_wgs84.index:
            geom = grid_wgs84.loc[grid_id].geometry
            if geom.geom_type == 'Polygon':
                coords = [[y, x] for x, y in geom.exterior.coords]
            else:
                half = 0.009
                coords = [
                    [clat - half, clon - half],
                    [clat - half, clon + half],
                    [clat + half, clon + half],
                    [clat + half, clon - half],
                    [clat - half, clon - half],
                ]
        else:
            # Fallback: 2 km approximate square
            half = 0.010
            coords = [
                [clat - half, clon - half],
                [clat - half, clon + half],
                [clat + half, clon + half],
                [clat + half, clon - half],
                [clat - half, clon - half],
            ]

        popup_html = (
            f"<div style='font-family:Arial;min-width:220px;padding:4px'>"
            f"<div style='background:{color};color:{'#111' if rank>=4 else '#fff'};"
            f"padding:6px 10px;border-radius:6px 6px 0 0;font-weight:700;font-size:14px'>"
            f"  #{rank} Predicted Location</div>"
            f"<div style='padding:8px 4px;line-height:1.7'>"
            f"  <b>Grid ID:</b> {grid_id}<br>"
            f"  <b>Probability:</b> <span style='color:{color};font-weight:700'>{prob:.2%}</span><br>"
            f"  <b>Centroid Lat:</b> {clat:.5f}°<br>"
            f"  <b>Centroid Lon:</b> {clon:.5f}°<br>"
            f"  <b>Cell Size:</b> 5 km × 5 km<br>"
            f"  <b>Risk Radius:</b> 5 km buffer"
            f"</div></div>"
        )

        folium.Polygon(
            locations=coords,
            color=color,
            weight=3,
            fill=True,
            fill_color=color,
            fill_opacity=opac,
            tooltip=f"<b>Rank #{rank}</b> — {grid_id} | Prob: <b>{prob:.1%}</b>",
            popup=folium.Popup(popup_html, max_width=260)
        ).add_to(fg_grids)

        # 5 km risk radius
        folium.Circle(
            location=[clat, clon],
            radius=5000,
            color=color,
            weight=1.8,
            fill=True,
            fill_color=color,
            fill_opacity=0.04,
            dash_array="8,5",
            tooltip=f"5 km alert radius — Rank #{rank} ({grid_id})"
        ).add_to(fg_radius)

        # Rank badge label
        txt_color = "#111111" if rank >= 4 else "#ffffff"
        badge_html = (
            f"<div style='background:{color};color:{txt_color};border:2.5px solid white;"
            f"border-radius:50%;width:30px;height:30px;display:flex;align-items:center;"
            f"justify-content:center;font-weight:800;font-size:13px;"
            f"box-shadow:0 2px 8px rgba(0,0,0,0.6);'>{rank}</div>"
        )
        folium.Marker(
            location=[clat, clon],
            icon=folium.DivIcon(html=badge_html, icon_size=(30, 30), icon_anchor=(15, 15)),
            tooltip=f"#{rank} {grid_id} ({prob:.1%})"
        ).add_to(fg_labels)

    # ── Trajectory lines: current → each predicted cell ───────────────────────
    for rank, info in sorted(rank_stats.items()):
        clat, clon = info['lat'], info['lon']
        color = RANK_COLORS[rank - 1] if rank <= 5 else "#FFFFFF"
        dist_deg = ((clat - cur_lat)**2 + (clon - cur_lon)**2) ** 0.5
        dist_km  = dist_deg * 111.0

        folium.PolyLine(
            locations=[[cur_lat, cur_lon], [clat, clon]],
            color=color,
            weight=max(1.0, 3.5 - (rank - 1) * 0.5),
            opacity=max(0.4, 0.9 - (rank - 1) * 0.1),
            dash_array="12,7",
            tooltip=f"Trajectory → Rank #{rank} | {dist_km:.1f} km away"
        ).add_to(fg_traj)

        folium.CircleMarker(
            location=[clat, clon],
            radius=6,
            color=color,
            fill=True, fill_color=color, fill_opacity=1.0, weight=2
        ).add_to(fg_traj)

    # ── Village markers ───────────────────────────────────────────────────────
    RISK_COLOR = {"HIGH": "#FF3B30", "MEDIUM": "#FF9500", "LOW": "#34C759"}
    RISK_EMOJI = {"HIGH": "⚠️", "MEDIUM": "⚡", "LOW": "✅"}

    if not villages_df.empty and 'village_lon_utm' in villages_df.columns:
        pts = gpd.GeoSeries(
            [Point(x, y) for x, y in zip(villages_df['village_lon_utm'], villages_df['village_lat_utm'])],
            crs="EPSG:32734"
        ).to_crs("EPSG:4326")

        for idx, (_, vrow) in enumerate(villages_df.iterrows()):
            vlon = pts.iloc[idx].x
            vlat = pts.iloc[idx].y
            risk = vrow.get('risk_level', 'LOW')
            vc   = RISK_COLOR.get(risk, "#AAAAAA")
            ve   = RISK_EMOJI.get(risk, "ℹ️")

            icon_html = (
                f"<div style='background:{vc};border:2px solid white;border-radius:6px;"
                f"padding:3px 5px;font-size:15px;box-shadow:0 2px 6px rgba(0,0,0,0.5)'>{ve}</div>"
            )
            vname = vrow.get('name', 'Unknown')
            vdist = vrow.get('distance_m', 0)
            vprob = vrow.get('probability_score', 0)

            popup_html = (
                f"<div style='font-family:Arial;min-width:200px;padding:4px'>"
                f"<div style='background:{vc};color:#111;padding:6px 10px;"
                f"border-radius:6px 6px 0 0;font-weight:700'>{ve} {vname}</div>"
                f"<div style='padding:8px 4px;line-height:1.7'>"
                f"  <b>Risk Level:</b> <span style='color:{vc};font-weight:700'>{risk}</span><br>"
                f"  <b>Distance to cell:</b> {vdist:.0f} m<br>"
                f"  <b>Associated Prob:</b> {vprob:.2%}<br>"
                f"  <b>Latitude:</b> {vlat:.5f}°<br>"
                f"  <b>Longitude:</b> {vlon:.5f}°"
                f"</div></div>"
            )
            folium.Marker(
                location=[vlat, vlon],
                icon=folium.DivIcon(html=icon_html, icon_size=(34, 30), icon_anchor=(17, 15)),
                tooltip=f"<b>{vname}</b> — {risk} RISK | {vdist:.0f} m from predicted cell",
                popup=folium.Popup(popup_html, max_width=240)
            ).add_to(fg_villages)

    # ── Historical Path (if provided) ─────────────────────────────────────────
    if historical_path is not None and len(historical_path) > 0:
        hist_coords = []
        for g_id in historical_path:
            if grid_wgs84 is not None and g_id in grid_wgs84.index:
                pt = grid_wgs84.loc[g_id].geometry.centroid
                hist_coords.append([pt.y, pt.x])
                
        # Connect historical path linearly up to the current position
        if len(hist_coords) > 0:
            hist_coords.append([cur_lat, cur_lon])
            folium.PolyLine(
                locations=hist_coords,
                color="#FFFFFF",
                weight=4,
                opacity=0.8,
                dash_array="5, 5",
                tooltip="🕰️ Historical Path (Last 24h)"
            ).add_to(fg_history)
            
            # Start marker
            folium.CircleMarker(
                location=hist_coords[0],
                radius=6,
                color="#FFFFFF",
                fill=True,
                fill_color="#000000",
                fill_opacity=1.0,
                weight=3,
                tooltip="Path Start"
            ).add_to(fg_history)

    # ── Add all feature groups ────────────────────────────────────────────────
    fg_grid_overlay.add_to(m)
    fg_grids.add_to(m)
    fg_traj.add_to(m)
    fg_radius.add_to(m)
    fg_history.add_to(m)
    fg_villages.add_to(m)
    fg_labels.add_to(m)
    fg_current.add_to(m)

    folium.LayerControl(collapsed=False, position="topright").add_to(m)

    # ── Info panel & legend ───────────────────────────────────────────────────
    top1       = rank_stats.get(1, {})
    top1_grid  = top1.get('grid_id', 'N/A')
    top1_prob  = top1.get('prob', 0)
    top1_lat   = top1.get('lat', 0)
    top1_lon   = top1.get('lon', 0)
    vcount     = len(villages_df) if not villages_df.empty else 0
    safe       = vcount == 0
    action_txt = "✅ SAFE — No settlements in path" if safe else f"⚠️ {vcount} VILLAGE(S) AT RISK"
    act_col    = "#34C759" if safe else "#FF3B30"
    now_ts     = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Build probability bar rows for each ranked prediction
    pred_table_rows = ""
    for r in range(1, 6):
        if r in rank_stats:
            rs = rank_stats[r]
            pct = rs['prob'] * 100
            bar_w = min(100, pct * 10)   # scale bar width (max 10% → 100px)
            dist_deg = ((rs['lat'] - cur_lat)**2 + (rs['lon'] - cur_lon)**2) ** 0.5
            dist_km  = dist_deg * 111.0
            txt_c = "#111" if r >= 4 else "#fff"
            pred_table_rows += f"""
            <div style='margin:6px 0'>
              <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:2px'>
                <span>
                  <span style='background:{RANK_COLORS[r-1]};color:{txt_c};
                    border-radius:50%;display:inline-block;width:20px;height:20px;
                    text-align:center;line-height:20px;font-weight:800;font-size:11px'>{r}</span>
                  &nbsp;<code style='font-size:11px;color:#ccc'>{rs['grid_id']}</code>
                </span>
                <span style='color:{RANK_COLORS[r-1]};font-weight:700;font-size:12px'>{pct:.2f}%</span>
              </div>
              <div style='background:rgba(255,255,255,0.08);border-radius:4px;height:6px;overflow:hidden'>
                <div style='background:{RANK_COLORS[r-1]};width:{bar_w:.0f}%;height:100%;border-radius:4px'></div>
              </div>
              <div style='font-size:10px;color:#666;margin-top:2px'>{rs['lat']:.4f}°, {rs['lon']:.4f}° &nbsp;|&nbsp; {dist_km:.1f} km away</div>
            </div>"""

    # Village summary rows
    village_rows_html = ""
    if not villages_df.empty:
        for _, vr in villages_df.iterrows():
            risk = vr.get('risk_level', 'LOW')
            vc = RISK_COLOR.get(risk, "#888")
            village_rows_html += (
                f"<div style='display:flex;justify-content:space-between;margin:3px 0;font-size:12px'>"
                f"<span style='color:#ccc'>{vr.get('name','Unknown')}</span>"
                f"<span style='color:{vc};font-weight:700'>{risk} ({vr.get('distance_m',0):.0f} m)</span>"
                f"</div>"
            )
    else:
        village_rows_html = "<div style='color:#666;font-size:12px;font-style:italic'>No villages within 5 km</div>"

    custom_html = f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
      .ew-panel {{
        position: fixed;
        top: 12px; left: 12px; z-index: 9999;
        background: rgba(12,12,22,0.95);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 16px;
        padding: 18px 20px;
        width: 310px;
        font-family: 'Inter', sans-serif;
        color: #f0f0f0;
        box-shadow: 0 12px 40px rgba(0,0,0,0.8);
        max-height: calc(100vh - 30px);
        overflow-y: auto;
      }}
      .ew-panel::-webkit-scrollbar {{ width: 4px; }}
      .ew-panel::-webkit-scrollbar-thumb {{ background: rgba(255,255,255,0.15); border-radius:2px; }}
      .ew-title {{ margin:0 0 2px; font-size:17px; font-weight:800; letter-spacing:0.3px; }}
      .ew-ts {{ font-size:10px; color:#555; margin-bottom:12px; font-family:'JetBrains Mono',monospace; }}
      .ew-section-title {{
        font-size:10px; font-weight:700; text-transform:uppercase;
        letter-spacing:1.2px; color:#555; margin:12px 0 6px;
        border-top:1px solid rgba(255,255,255,0.07); padding-top:10px;
      }}
      .ew-stat {{ display:flex;justify-content:space-between;align-items:center;margin:5px 0;font-size:12.5px; }}
      .ew-stat .label {{ color:#888; }}
      .ew-stat .val {{ color:#fff;font-weight:600;font-family:'JetBrains Mono',monospace;font-size:12px; }}
      .ew-action {{
        margin-top:10px; padding:10px 14px; border-radius:10px;
        font-size:13px; font-weight:700;
        background: rgba(255,255,255,0.05);
        border-left: 4px solid {act_col};
        color: {act_col}; text-align:center;
      }}
    </style>
    <div class="ew-panel">
      <div class="ew-title">🐘 Early Warning System</div>
      <div class="ew-ts">Generated: {now_ts} IST &nbsp;|&nbsp; Model: Random Forest</div>

      <div class="ew-section-title">📍 Current State</div>
      <div class="ew-stat"><span class="label">Elephant ID</span><span class="val">E-1</span></div>
      <div class="ew-stat"><span class="label">Grid Cell</span><span class="val">{current_grid}</span></div>
      <div class="ew-stat"><span class="label">Position</span><span class="val">{cur_lat:.4f}°, {cur_lon:.4f}°</span></div>
      <div class="ew-stat"><span class="label">Horizon</span><span class="val">48 hours</span></div>

      <div class="ew-section-title">🎯 Top Prediction</div>
      <div class="ew-stat"><span class="label">Grid Cell</span><span class="val">{top1_grid}</span></div>
      <div class="ew-stat"><span class="label">Confidence</span><span class="val" style="color:{RANK_COLORS[0]}">{top1_prob:.2%}</span></div>
      <div class="ew-stat"><span class="label">Location</span><span class="val">{top1_lat:.4f}°, {top1_lon:.4f}°</span></div>

      <div class="ew-action">{action_txt}</div>

      <div class="ew-section-title">📊 Ranked Predictions</div>
      {pred_table_rows}

      <div class="ew-section-title">🏘️ Villages ({vcount} at risk)</div>
      {village_rows_html}

      <div class="ew-section-title">🗺️ Map Guide</div>
      <div style='font-size:11px;color:#555;line-height:1.8'>
        <span style='color:#FF3B30'>■</span> #1 &nbsp;
        <span style='color:#FF9500'>■</span> #2 &nbsp;
        <span style='color:#FFCC00'>■</span> #3 &nbsp;
        <span style='color:#34C759'>■</span> #4 &nbsp;
        <span style='color:#007AFF'>■</span> #5<br>
        Dashed lines = trajectory paths<br>
        Dashed circles = 5 km alert zones<br>
        Toggle layers via panel (top-right)
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(custom_html))

    m.save('alert_map.html')
    print(f"[{datetime.now().isoformat()}] Map saved to alert_map.html")



if __name__ == "__main__":
    pass

