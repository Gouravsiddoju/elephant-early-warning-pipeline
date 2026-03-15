# -*- coding: utf-8 -*-
import sys
import io
# Force UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

"""
Multi-Elephant Prediction Viewer
=================================
Loads real GPS data from feature_matrix.csv, selects multiple elephants,
runs the trained LSTM model for each, and generates a single interactive
Folium map showing historical paths + predicted future locations.

Run from: early_warning_pipeline/
    python multi_elephant_prediction.py
"""

import os
import torch
import joblib
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import HeatMap
from shapely.geometry import Point
from datetime import datetime
import pyproj

# Local imports
from model_trainer import ElephantLSTM
from predictor import predict_next_grid
from grid_builder import build_grid

# ─── Constants ────────────────────────────────────────────────────────────────
BASE          = os.path.dirname(os.path.abspath(__file__))
FEATURE_CSV   = os.path.join(BASE, 'feature_matrix.csv')
MODEL_PT      = os.path.join(BASE, 'elephant_lstm.pt')
SCALER_PKL    = os.path.join(BASE, 'scaler.pkl')
LE_PKL        = os.path.join(BASE, 'label_encoder.pkl')
FEATNAMES_PKL = os.path.join(BASE, 'feature_names.pkl')
CENTROIDS_CSV = os.path.join(BASE, 'grid_centroids.csv')
OUTPUT_MAP    = os.path.join(BASE, 'multi_elephant_map.html')

SEQ_LEN       = 10
N_ELEPHANTS   = 8        # How many elephants to pick
MIN_HISTORY   = SEQ_LEN  # Must have at least this many GPS steps

# Distinct colors for up to 10 elephants (colorblind-friendly palette)
ELEPHANT_COLORS = [
    "#FF6B6B", "#4ECDC4", "#FFE66D", "#A8E6CF",
    "#FF8B94", "#A0C4FF", "#BDB2FF", "#FFC6FF",
    "#FFAFCC", "#80B3FF"
]

def rebuild_grid():
    STUDY_LON_MIN, STUDY_LON_MAX = 23.0, 28.0
    STUDY_LAT_MIN, STUDY_LAT_MAX = -22.0, -17.0
    proj_wgs = pyproj.CRS('EPSG:4326')
    proj_utm = pyproj.CRS('EPSG:32734')
    transformer = pyproj.Transformer.from_crs(proj_wgs, proj_utm, always_xy=True)
    min_x, min_y = transformer.transform(STUDY_LON_MIN, STUDY_LAT_MIN)
    max_x, max_y = transformer.transform(STUDY_LON_MAX, STUDY_LAT_MAX)
    return build_grid((min_x, min_y, max_x, max_y), cell_size_m=5000)


# Grid step constants derived from grid_centroids.csv
# R0050_C0108 = (28.113°, -19.264°)  →  row+1 = -0.04488° lat, col+1 = +0.04488° lon
# (5km cells in UTM 34S projected back to degrees at ~-18° lat)
_GRID_LAT_ORIGIN = -17.0 + 0.02244   # lat at row=0 centroid (approx top of study area)
_GRID_LON_ORIGIN = 23.0  + 0.02244   # lon at col=0 centroid (approx left of study area)
_GRID_LAT_STEP   = -0.04488           # degrees latitude per row increment (south)
_GRID_LON_STEP   =  0.04488           # degrees longitude per col increment (east)

def grid_id_to_latlon(grid_id: str):
    """
    Derive (lat, lon) from a grid_id string like 'R0062_C0116' mathematically.
    Returns (lat, lon) or (None, None) if the format is invalid.
    Works for ANY grid_id even if it's not in the grid_centroids.csv lookup.
    """
    try:
        parts = grid_id.split('_')
        row = int(parts[0][1:])   # strip 'R' prefix
        col = int(parts[1][1:])   # strip 'C' prefix
        lat = _GRID_LAT_ORIGIN + row * _GRID_LAT_STEP
        lon = _GRID_LON_ORIGIN + col * _GRID_LON_STEP
        return float(lat), float(lon)
    except Exception:
        return None, None


def load_artifacts():
    print(f"[{datetime.now().isoformat()}] Loading model artifacts...")
    scaler        = joblib.load(SCALER_PKL)
    label_encoder = joblib.load(LE_PKL)
    features      = joblib.load(FEATNAMES_PKL)
    num_classes   = len(label_encoder.classes_)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[{datetime.now().isoformat()}] Using device: {device}")

    model = ElephantLSTM(
        input_dim=len(features), hidden_dim=128,
        num_layers=2, output_dim=num_classes
    )
    model.load_state_dict(torch.load(MODEL_PT, map_location=device))
    model.to(device)
    model.eval()
    print(f"[{datetime.now().isoformat()}] Model loaded ({num_classes} classes, {len(features)} features).")
    return model, scaler, label_encoder, features


def select_elephants(df: pd.DataFrame, n: int = N_ELEPHANTS):
    """Pick the N elephants with the most GPS steps (most data = best predictions)."""
    counts = df.groupby('elephant_id').size().sort_values(ascending=False)
    eligible = counts[counts >= MIN_HISTORY].index.tolist()
    selected = eligible[:n]
    print(f"[{datetime.now().isoformat()}] Selected {len(selected)} elephants: {selected}")
    return selected


def get_elephant_sequence(df: pd.DataFrame, eid, features: list):
    """
    For a given elephant ID, return:
    - input_seq: list of dicts for the last SEQ_LEN steps (for prediction)
    - history_grids: list of grid_ids for the last 20 steps (for map trail)
    - centroid for current location (from grid_centroids)
    """
    grp = df[df['elephant_id'] == eid].sort_values('Date_Time')
    if len(grp) < MIN_HISTORY:
        return None, None

    # Return the last 25 rows for trail building + last SEQ_LEN for prediction
    history_rows = grp.tail(25).copy().reset_index(drop=True)

    # Sequence for prediction (last SEQ_LEN raw rows, feature cols only)
    exclude_cols = ['elephant_id', 'Date_Time', 'from_grid', 'to_grid',
                    'target', 'grid_centroid_lon', 'grid_centroid_lat']
    seq_rows = grp.tail(SEQ_LEN)
    input_seq = seq_rows[[c for c in grp.columns if c not in exclude_cols]].to_dict('records')

    return input_seq, history_rows


def dead_reckon_path(history_rows: pd.DataFrame, start_lat: float, start_lon: float) -> list:
    """
    Reconstruct actual GPS waypoints from step_dist_m + turning_angle using
    dead reckoning from a known starting centroid. Returns [[lat, lon], ...].
    This is necessary because elephants often stay in the same 5km grid cell,
    making grid-centroid-based paths appear as a single unmoving dot.
    """
    DEG_PER_M = 1.0 / 111_000.0
    coords = [[start_lat, start_lon]]
    heading = 0.0  # initial heading (north)

    for _, row in history_rows.iterrows():
        dist_m = float(row.get('step_dist_m', 0) or 0)
        angle  = float(row.get('turning_angle', 0) or 0)
        heading += angle

        dlat = dist_m * np.cos(heading) * DEG_PER_M
        dlon = dist_m * np.sin(heading) * DEG_PER_M / max(np.cos(np.radians(coords[-1][0])), 0.01)

        new_lat = coords[-1][0] + dlat
        new_lon = coords[-1][1] + dlon
        coords.append([new_lat, new_lon])

    return coords


def make_map(elephant_results: list, grid_wgs84, centroids_map: dict) -> folium.Map:
    """Build the full interactive Folium map with all elephants."""

    # Figure out a good center from all current positions
    all_lats, all_lons = [], []
    for er in elephant_results:
        if er.get('cur_lat') and er.get('cur_lon'):
            all_lats.append(er['cur_lat'])
            all_lons.append(er['cur_lon'])

    center_lat = float(np.mean(all_lats)) if all_lats else -18.5
    center_lon = float(np.mean(all_lons)) if all_lons else 24.5

    m = folium.Map(location=[center_lat, center_lon], zoom_start=9, tiles=None)

    # Basemaps
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr='&copy; <a href="https://carto.com/">CARTO</a>',
        name="🌑 Dark (CARTO)", max_zoom=19
    ).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="ESRI", name="🛰️ Satellite (ESRI)", max_zoom=19
    ).add_to(m)
    folium.TileLayer(tiles="OpenStreetMap", name="🗺️ Street (OSM)").add_to(m)

    # Feature groups
    fg_positions  = folium.FeatureGroup(name="🐘 Current Positions",    show=True)
    fg_history    = folium.FeatureGroup(name="🕰️ Historical Paths",     show=True)
    fg_predicted  = folium.FeatureGroup(name="🎯 Predicted Locations",  show=True)
    fg_traj       = folium.FeatureGroup(name="➡️ Prediction Vectors",   show=True)

    legend_rows = ""

    for i, er in enumerate(elephant_results):
        eid   = er['elephant_id']
        color = ELEPHANT_COLORS[i % len(ELEPHANT_COLORS)]
        cur_lat = er.get('cur_lat')
        cur_lon = er.get('cur_lon')
        preds   = er.get('predictions')       # pd.DataFrame
        history = er.get('history_coords', []) # [[lat,lon], ...]

        if cur_lat is None:
            continue

        # ── Current Position marker ──────────────────────────────────────────
        elephant_icon = (
            f"<div style='background:{color};border:3px solid white;border-radius:50%;"
            f"width:36px;height:36px;display:flex;align-items:center;justify-content:center;"
            f"font-size:18px;box-shadow:0 0 14px {color}88;'>"
            f"🐘</div>"
        )
        folium.Marker(
            location=[cur_lat, cur_lon],
            icon=folium.DivIcon(html=elephant_icon, icon_size=(36, 36), icon_anchor=(18, 18)),
            tooltip=f"<b>🐘 Elephant {eid}</b><br>Current: {cur_lat:.4f}°, {cur_lon:.4f}°",
            popup=folium.Popup(
                f"<div style='font-family:Arial;min-width:200px'>"
                f"<b style='font-size:14px;color:{color}'>🐘 Elephant ID: {eid}</b><hr style='margin:4px 0'>"
                f"<b>Current Grid:</b> {er.get('current_grid','?')}<br>"
                f"<b>Lat:</b> {cur_lat:.5f}°<br><b>Lon:</b> {cur_lon:.5f}°"
                f"</div>",
                max_width=220
            )
        ).add_to(fg_positions)

        # -- Historical Path trail (solid, thick, highly visible) --
        if len(history) >= 2:
            folium.PolyLine(
                locations=history,
                color=color,
                weight=5,
                opacity=0.95,
                tooltip=f"Elephant {eid} — Historical Path ({len(history)} steps)"
            ).add_to(fg_history)
            # Trail start dot (hollow circle)
            folium.CircleMarker(
                location=history[0],
                radius=6,
                color=color,
                fill=True,
                fill_color="#000",
                fill_opacity=1.0,
                weight=3,
                tooltip=f"Path start — Elephant {eid}"
            ).add_to(fg_history)
            # Small waypoint dots along path
            for coord in history[1:-1:3]:  # every 3rd point
                folium.CircleMarker(
                    location=coord,
                    radius=3,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.7,
                    weight=1
                ).add_to(fg_history)

        # ── Predicted future locations ────────────────────────────────────────
        if preds is not None and not preds.empty:
            top1 = preds.iloc[0]
            for _, pred_row in preds.iterrows():
                rank = int(pred_row.get('rank', 1))
                plat = pred_row.get('centroid_lat')
                plon = pred_row.get('centroid_lon')
                prob = float(pred_row.get('probability', 0))
                pgrid = str(pred_row.get('grid_id', '?'))

                if pd.isna(plat) or pd.isna(plon):
                    continue

                # Predicted polygon approximation (2km half-side)
                half = 0.023
                coords = [
                    [plat - half, plon - half], [plat - half, plon + half],
                    [plat + half, plon + half], [plat + half, plon - half],
                    [plat - half, plon - half]
                ]
                opacity = max(0.10, 0.50 - (rank - 1) * 0.08)
                folium.Polygon(
                    locations=coords,
                    color=color,
                    weight=2,
                    fill=True,
                    fill_color=color,
                    fill_opacity=opacity,
                    tooltip=(
                        f"<b>🐘 {eid}</b> — Rank #{rank}<br>"
                        f"Grid: <code>{pgrid}</code><br>"
                        f"Probability: <b>{prob:.2%}</b>"
                    ),
                    popup=folium.Popup(
                        f"<div style='font-family:Arial;min-width:190px'>"
                        f"<div style='background:{color};color:#111;padding:6px 10px;"
                        f"border-radius:6px 6px 0 0;font-weight:700'>"
                        f"Elephant {eid} — Rank #{rank}</div>"
                        f"<div style='padding:8px 4px;line-height:1.7'>"
                        f"<b>Grid:</b> {pgrid}<br>"
                        f"<b>Prob:</b> <span style='color:{color};font-weight:700'>{prob:.2%}</span><br>"
                        f"<b>Coords:</b> {plat:.4f}°, {plon:.4f}°</div></div>",
                        max_width=220
                    )
                ).add_to(fg_predicted)

                # Trajectory arrow — solid bold line from current → top prediction
                if rank == 1:
                    # White glow outline for contrast on dark basemap
                    folium.PolyLine(
                        locations=[[cur_lat, cur_lon], [plat, plon]],
                        color='white',
                        weight=6,
                        opacity=0.25,
                    ).add_to(fg_traj)
                    folium.PolyLine(
                        locations=[[cur_lat, cur_lon], [plat, plon]],
                        color=color,
                        weight=4,
                        opacity=1.0,
                        tooltip=f"Elephant {eid} - Most likely next location ({prob:.1%})"
                    ).add_to(fg_traj)
                    # Arrowhead at destination
                    folium.CircleMarker(
                        location=[plat, plon],
                        radius=8,
                        color='white',
                        fill=True,
                        fill_color=color,
                        fill_opacity=1.0,
                        weight=2,
                        tooltip=f"Top prediction: Elephant {eid} ({prob:.2%})"
                    ).add_to(fg_traj)

        # Legend row
        top1_prob = float(preds.iloc[0]['probability']) if preds is not None and not preds.empty else 0.0
        top1_grid = str(preds.iloc[0]['grid_id']) if preds is not None and not preds.empty else "N/A"
        legend_rows += (
            f"<div style='display:flex;align-items:center;gap:10px;margin:6px 0;"
            f"padding:6px 8px;background:rgba(255,255,255,0.04);border-radius:8px;"
            f"border-left:3px solid {color}'>"
            f"<span style='font-size:20px'>🐘</span>"
            f"<div style='flex:1'>"
            f"<div style='font-weight:700;font-size:13px;color:{color}'>Elephant {eid}</div>"
            f"<div style='font-size:11px;color:#888'>Top: <code style='color:#ccc'>{top1_grid}</code>"
            f" &nbsp;|&nbsp; <span style='color:{color}'>{top1_prob:.2%}</span></div>"
            f"</div></div>"
        )

    fg_history.add_to(m)
    fg_traj.add_to(m)
    fg_predicted.add_to(m)
    fg_positions.add_to(m)
    folium.LayerControl(collapsed=False, position="topright").add_to(m)

    # ── Panel ─────────────────────────────────────────────────────────────────
    now_ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    panel_html = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono&display=swap');
    .mep-panel {{
        position: fixed; top: 12px; left: 12px; z-index: 9999;
        background: rgba(9,9,18,0.96);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 16px 18px;
        width: 300px;
        font-family: 'Inter', sans-serif;
        color: #f0f0f0;
        box-shadow: 0 12px 40px rgba(0,0,0,0.85);
        max-height: calc(100vh - 30px);
        overflow-y: auto;
    }}
    .mep-panel::-webkit-scrollbar {{ width: 4px; }}
    .mep-panel::-webkit-scrollbar-thumb {{ background: rgba(255,255,255,0.12); border-radius:2px; }}
    .mep-title {{ margin:0 0 2px; font-size:16px; font-weight:800; }}
    .mep-ts {{ font-size:10px; color:#555; margin-bottom:12px; font-family:'JetBrains Mono',monospace; }}
    .mep-section {{ font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:1.2px;
        color:#555; margin:12px 0 6px; border-top:1px solid rgba(255,255,255,0.06); padding-top:10px; }}
    .mep-guide {{ font-size:11px; color:#555; line-height:1.9; }}
    </style>
    <div class="mep-panel">
      <div class="mep-title">🐘 Multi-Elephant Tracker</div>
      <div class="mep-ts">Generated: {now_ts} CAT &nbsp;|&nbsp; Model: LSTM</div>
      <div class="mep-section">📊 {len(elephant_results)} Elephants Tracked</div>
      {legend_rows}
      <div class="mep-section">🗺️ Map Guide</div>
      <div class="mep-guide">
        🐘 Marker = Current Position<br>
        — Dashed trail = Historical path<br>
        ⬜ Shaded boxes = Predicted cells<br>
        ➡️ Arrow = #1 Most likely move<br>
        Toggle layers via panel (top-right)
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(panel_html))
    return m


def main():
    print(f"\n{'='*60}")
    print("  MULTI-ELEPHANT PREDICTION VIEWER")
    print(f"{'='*60}\n")

    # Load model
    model, scaler, label_encoder, features = load_artifacts()

    # Load grid
    print(f"[{datetime.now().isoformat()}] Building spatial grid...")
    grid_gdf  = rebuild_grid()
    grid_wgs84 = grid_gdf.to_crs("EPSG:4326").set_index('grid_id')
    print(f"[{datetime.now().isoformat()}] Grid ready ({len(grid_wgs84)} cells).")

    # Load centroids lookup
    centroid_map = {}
    if os.path.exists(CENTROIDS_CSV):
        cd_df = pd.read_csv(CENTROIDS_CSV)
        centroid_map = cd_df.set_index('grid_id').to_dict('index')

    # Load feature matrix
    print(f"[{datetime.now().isoformat()}] Loading feature matrix...")
    df = pd.read_csv(FEATURE_CSV)
    df['Date_Time'] = pd.to_datetime(df['Date_Time'], utc=True, errors='coerce')

    # Filter to valid classes only
    class_counts  = df['to_grid'].value_counts()
    valid_classes = class_counts[class_counts >= 10].index
    df = df[df['to_grid'].isin(valid_classes)].copy()
    print(f"[{datetime.now().isoformat()}] Dataset: {len(df)} rows, "
          f"{df['elephant_id'].nunique()} unique elephants after filtering.")

    # Select elephants
    selected_eids = select_elephants(df)

    # Run predictions for each elephant
    elephant_results = []
    for i, eid in enumerate(selected_eids):
        print(f"\n[{datetime.now().isoformat()}] Processing Elephant {eid} ({i+1}/{len(selected_eids)})...")

        input_seq, history_grids = get_elephant_sequence(df, eid, features)
        if input_seq is None:
            print(f"  Skipping — not enough data.")
            continue

        current_grid = history_grids['from_grid'].iloc[-1] if not history_grids.empty else None

        # Resolve current (lat, lon) — walk backwards through history to find the
        # most recent grid that is actually in our grid index (dead reckoning can
        # push the last grid ID out of bounds, e.g. R0062_C0116 may not exist)
        cur_lat, cur_lon = None, None
        resolved_grid = None
        all_grids = history_grids['from_grid'].tolist()
        for g in reversed(all_grids):
            if g in grid_wgs84.index:
                centroid = grid_wgs84.loc[g].geometry.centroid
                cur_lon, cur_lat = float(centroid.x), float(centroid.y)
                resolved_grid = g
                break
            elif g in centroid_map:
                cur_lon = float(centroid_map[g].get('centroid_lon', 0))
                cur_lat = float(centroid_map[g].get('centroid_lat', 0))
                resolved_grid = g
                break
            else:
                lat_m, lon_m = grid_id_to_latlon(g)
                if lat_m is not None:
                    cur_lat, cur_lon = lat_m, lon_m
                    resolved_grid = g
                    break

        if cur_lat is None:
            print(f"  [WARN] No resolvable grid found for Elephant {eid}, skipping.")
            continue

        if resolved_grid != current_grid:
            print(f"  [INFO] Grid {current_grid} not in index, using nearest: {resolved_grid}")
            current_grid = resolved_grid

        # Resolve history coords using dead reckoning from earliest resolvable grid
        history_coords = []
        start_lat, start_lon = cur_lat, cur_lon
        for g in history_grids['from_grid'].tolist():
            if g in grid_wgs84.index:
                pt = grid_wgs84.loc[g].geometry.centroid
                start_lat, start_lon = float(pt.y), float(pt.x)
                break
            elif g in centroid_map:
                start_lat = float(centroid_map[g].get('centroid_lat', cur_lat))
                start_lon = float(centroid_map[g].get('centroid_lon', cur_lon))
                break
        history_coords = dead_reckon_path(history_grids, start_lat, start_lon)

        # Run LSTM prediction
        try:
            preds_df = predict_next_grid(
                model, scaler, label_encoder, input_seq,
                feature_names_path=FEATNAMES_PKL,
                centroids_path=CENTROIDS_CSV
            )
            print(f"  [OK] Top prediction: {preds_df.iloc[0]['grid_id']} "
                  f"({preds_df.iloc[0]['probability']:.2%})")
        except Exception as e:
            print(f"  [WARN] Prediction failed: {e}")
            preds_df = None

        elephant_results.append({
            'elephant_id'  : eid,
            'current_grid' : current_grid,
            'cur_lat'      : cur_lat,
            'cur_lon'      : cur_lon,
            'history_coords': history_coords,
            'predictions'  : preds_df,
        })

    if not elephant_results:
        print("No elephants processed. Check your data and model files.")
        sys.exit(1)

    # Build map
    print(f"\n[{datetime.now().isoformat()}] Building map for {len(elephant_results)} elephants...")
    m = make_map(elephant_results, grid_wgs84, centroid_map)
    m.save(OUTPUT_MAP)
    print(f"\n[{datetime.now().isoformat()}] Map saved -> {OUTPUT_MAP}")
    print("Open it in your browser to explore!")


if __name__ == '__main__':
    main()
