"""
prediction_service.py
Modular service for loading the LSTM model and generating 
elephant dashboard data for real-time and batch usage.
"""

import os, sys, json, torch, joblib, pyproj
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from model_trainer import ElephantLSTM
from predictor import predict_next_grid
from grid_builder import build_grid

# Feature groups for reasoning
HUMAN_IMPACT_FEATURES = ['village_distance_m', 'cropland_pct', 'village_risk', 'crop_attraction']
ENV_FEATURES = ['ndvi', 'rainfall', 'landcover', 'temp']
MOVEMENT_FEATURES = ['step_dist_m', 'turning_angle', 'cos_angle', 'sin_angle', 'log_step_dist']

# Constants for grid resolution
DEG_PER_M = 1.0 / 111_000.0
_GRID_LAT_ORIGIN = -17.0 + 0.02244
_GRID_LON_ORIGIN = 23.0  + 0.02244
_GRID_LAT_STEP   = -0.04488
_GRID_LON_STEP   =  0.04488

class PredictionService:
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load resources
        self.scaler = joblib.load(os.path.join(base_path, 'scaler.pkl'))
        self.label_encoder = joblib.load(os.path.join(base_path, 'label_encoder.pkl'))
        self.features = joblib.load(os.path.join(base_path, 'feature_names.pkl'))
        self.num_classes = len(self.label_encoder.classes_)
        
        # Initialize and load model
        self.model = ElephantLSTM(
            input_dim=len(self.features), 
            hidden_dim=128, 
            num_layers=2, 
            output_dim=self.num_classes
        )
        self.model.load_state_dict(torch.load(os.path.join(base_path, 'elephant_lstm.pt'), map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # Build grid
        proj = pyproj.Transformer.from_crs('EPSG:4326','EPSG:32734',always_xy=True)
        min_x, min_y = proj.transform(23.0, -22.0)
        max_x, max_y = proj.transform(28.0, -17.0)
        grid_gdf = build_grid((min_x, min_y, max_x, max_y), cell_size_m=5000)
        self.grid_wgs84 = grid_gdf.to_crs('EPSG:4326').set_index('grid_id')
        
        self.centroid_map = {}
        centroid_path = os.path.join(base_path, 'grid_centroids.csv')
        if os.path.exists(centroid_path):
            self.centroid_map = pd.read_csv(centroid_path).set_index('grid_id').to_dict('index')

    def resolve_latlon(self, grid_id):
        if grid_id in self.grid_wgs84.index:
            c = self.grid_wgs84.loc[grid_id].geometry.centroid
            return float(c.y), float(c.x)
        if grid_id in self.centroid_map:
            return float(self.centroid_map[grid_id]['centroid_lat']), \
                   float(self.centroid_map[grid_id]['centroid_lon'])
        
        # Fallback grid math
        try:
            parts = grid_id.split('_')
            row = int(parts[0][1:])
            col = int(parts[1][1:])
            return float(_GRID_LAT_ORIGIN + row * _GRID_LAT_STEP), \
                   float(_GRID_LON_ORIGIN + col * _GRID_LON_STEP)
        except:
            return 0, 0

    def generate_reasoning(self, context, grid_id, prob):
        """
        Translates raw feature values and prediction results into human-readable 
        reasoning strings.
        """
        reasons = []
        
        # 1. Environmental Analysis
        ndvi = context.get('ndvi', 0)
        rainfall = context.get('rainfall', 0)
        if ndvi > 0.4:
            reasons.append(f"High vegetation density (NDVI: {ndvi:.2f}) observed in this region.")
        elif ndvi > 0.2:
            reasons.append("Moderate foraging habitat detected.")
            
        if rainfall > 50:
            reasons.append(f"Recent high rainfall ({rainfall:.1f}mm) attracts movement toward water sources.")

        # 2. Human-Elephant Conflict Markers
        village_dist = context.get('village_distance_m', 10000)
        cropland = context.get('cropland_pct', 0)
        
        if cropland > 10:
            reasons.append(f"High cropland concentration ({cropland:.1f}%) detected.")
        if village_dist < 1500:
            reasons.append(f"Proximity to human settlements ({village_dist/1000.0:.1f} km) increases interaction risk.")
        
        # 3. Kinematic drivers
        speed = context.get('step_dist_m', 0)
        if speed > 2000:
            reasons.append("High movement velocity suggests migratory behavior.")
        elif speed < 500:
            reasons.append("Low step distance suggests localized foraging or resting.")

        # 4. Confidence / Probability
        if prob > 0.5:
            reasons.append("High model confidence based on strong historical kinematic patterns.")
        
        # Ensure we always have something
        if not reasons:
            reasons.append("Predicted based on baseline historical migration patterns for this grid.")
            
        return reasons

    def get_corrected_path(self, history_rows, current_lat, current_lon):
        """
        Fixes GPS drift by forcing the dead-reckoned path to end exactly at the 
        current known lat/lon (the 'Anchor').
        """
        coords = [[round(current_lat, 5), round(current_lon, 5)]]
        heading = 0.0
        
        # Dead reckon backwards/forwards to get relative shape
        # But for simpler 'snapping', we just generate normally and then apply a translation
        raw_path = [[0.0, 0.0]]
        for _, row in history_rows.iterrows():
            dist_m = float(row.get('step_dist_m', 0) or 0)
            angle = float(row.get('turning_angle', 0) or 0)
            heading += angle
            dlat = dist_m * np.cos(heading) * DEG_PER_M
            dlon = dist_m * np.sin(heading) * DEG_PER_M / max(np.cos(np.radians(current_lat)), 0.01)
            raw_path.append([raw_path[-1][0] + dlat, raw_path[-1][1] + dlon])
        
        # Last point in raw_path should be snapped to current_lat/lon
        last_p = raw_path[-1]
        lat_diff = current_lat - last_p[0]
        lon_diff = current_lon - last_p[1]
        
        corrected = [[round(p[0] + lat_diff, 5), round(p[1] + lon_diff, 5)] for p in raw_path]
        return corrected

    def generate_dashboard_data(self, df: pd.DataFrame, n_elephants=8):
        df['Date_Time'] = pd.to_datetime(df['Date_Time'], utc=True, errors='coerce')
        counts = df.groupby('elephant_id').size().sort_values(ascending=False)
        selected = counts[counts >= 10].index[:n_elephants].tolist()
        
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        exclude_cols = {'elephant_id','Date_Time','from_grid','to_grid','target','grid_centroid_lon','grid_centroid_lat'}
        
        elephants_out, preds_map, movement_map, history_map, alerts_out = [], {}, {}, {}, []

        for i, eid in enumerate(selected):
            eid_str = f"E-{eid}"
            grp = df[df['elephant_id'] == eid].sort_values('Date_Time')
            rows = grp.tail(25).copy().reset_index(drop=True)
            
            # Current Lat/Lon (Snapped to grid)
            cur_grid = rows.iloc[-1]['from_grid']
            cur_lat, cur_lon = self.resolve_latlon(cur_grid)
            
            # Predictions
            seq_rows = grp.tail(10)
            input_seq = seq_rows[[c for c in grp.columns if c not in exclude_cols]].to_dict('records')
            preds_df, context = predict_next_grid(self.model, self.scaler, self.label_encoder, input_seq, 
                                         feature_names_path=os.path.join(self.base_path, 'feature_names.pkl'),
                                         centroids_path=os.path.join(self.base_path, 'grid_centroids.csv'))
            
            preds_out = []
            for _, pr in preds_df.iterrows():
                plat, plon = self.resolve_latlon(str(pr['grid_id']))
                conf = round(float(pr['probability'])*100, 2)
                preds_out.append({
                    "rank": int(pr['rank']),
                    "gridCell": str(pr['grid_id']),
                    "confidence": conf,
                    "location": {"lat": round(plat, 5), "lng": round(plon, 5)},
                    "distanceKm": round(((plat-cur_lat)**2 + (plon-cur_lon)**2)**0.5 * 111.0, 1),
                    "reasoning": self.generate_reasoning(context, str(pr['grid_id']), float(pr['probability']))
                })

            # History with correction
            history_map[eid_str] = self.get_corrected_path(rows, cur_lat, cur_lon)
            
            # Status and alert
            top_conf = preds_out[0]["confidence"] if preds_out else 0
            status = "safe" if top_conf > 10.0 else "warning"
            
            elephants_out.append({
                "id": eid_str, "name": f"Elephant {eid}", "gridCell": cur_grid,
                "position": {"lat": round(cur_lat, 5), "lng": round(cur_lon, 5)},
                "horizon": 48, "model": "LSTM (PyTorch)", "lastUpdated": now_str, "status": status
            })
            preds_map[eid_str] = preds_out
            movement_map[eid_str] = [{"hour": r['Date_Time'].strftime("%H:%M"), "distance": round(float(r.get('step_dist_m', 0))/1000.0, 3)} for _, r in rows.iterrows()]
            
            alerts_out.append({
                "id": f"api_{eid_str}", "timestamp": now_str, "type": "prediction",
                "severity": "low" if status == "safe" else "medium",
                "message": f"Elephant {eid} predicted to reach {preds_out[0]['gridCell']} with {top_conf}% confidence.",
                "elephantId": eid_str
            })

        return {
            "generatedAt": now_str, "elephants": elephants_out, "predictionsMap": preds_map,
            "villagesMap": {e["id"]: [] for e in elephants_out}, "alertEvents": alerts_out,
            "movementMap": movement_map, "historyMap": history_map
        }
