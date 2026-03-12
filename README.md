# 🐘 Elephant Movement Prediction & Early Warning Pipeline

A production-quality AI pipeline that learns from historical elephant GPS data to **predict herd movement** and **generate early warnings** for at-risk villages — helping prevent human-elephant conflict in Botswana.

---

## 🗺️ Overview

```
Real GPS Data (Dryad) + OSM Botswana + Google Earth Engine
              ↓
    Spatial Grid (2km cells, UTM 34S)
              ↓
  Feature Engineering (5 Dataset Layers)
              ↓
   Random Forest Classifier (Multi-class)
              ↓
  Top-5 Grid Cell Predictions (48h window)
              ↓
   Village Risk Alert + Interactive Map
```

**Study Region:** Botswana (~23–28°E, -22 to -17°S)  
**Period:** April 2014 – December 2015 (Dryad dataset)  
**Individuals:** 11 GPS-tracked elephants  

---

## 📁 Pipeline Architecture

```
Stage 1  data_loader.py       → Load & validate Dryad CSV
Stage 2  grid_builder.py      → Build 2km UTM spatial grid, compute transitions
Stage 3  gee_extractor.py     → NDVI, rainfall, land cover via Google Earth Engine
Stage 4  human_features.py    → Village distances, road density, cropland (OSM)
Stage 5  memory_features.py   → Repeat transitions, site fidelity (no data leakage)
Stage 6  feature_matrix.py    → Merge all 5 layers, engineer 29 features
Stage 7  model_trainer.py     → Train RandomForest, chronological 80/20 split
Stage 8  predictor.py         → Live prediction: Top-5 next grid cells + probs
Stage 9  alert_engine.py      → 5km village buffer, risk levels, Folium map
Stage 10 main.py              → End-to-end orchestrator
```

---

## 🗂️ Data Sources

| Dataset | Source | Description |
|---------|--------|-------------|
| GPS movements | [Dryad doi:10.5061/dryad.dr7sqv9v9](https://doi.org/10.5061/dryad.dr7sqv9v9) | 213K hourly fixes, 11 elephants |
| Satellite imagery | Google Earth Engine | MODIS NDVI, CHIRPS rainfall, land cover |
| Human geography | OpenStreetMap Botswana | Villages, roads, cropland polygons |

---

## ⚙️ Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/elephant-early-warning-pipeline.git
cd elephant-early-warning-pipeline

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Authenticate Google Earth Engine for satellite features
earthengine authenticate

# 4. Place your data files:
#    - doi_10_5061_dryad_dr7sqv9v9__v20200116/ElephantsData_ano.csv
#    - botswana-260310-free.shp/  (Botswana OSM shapefiles folder)

# 5. Run the full pipeline
python main.py
```

---

## 📦 Output Files

| File | Description |
|------|-------------|
| `feature_matrix.csv` | 29-feature merged training dataset |
| `elephant_model.pkl` | Trained Random Forest model |
| `scaler.pkl` | Fitted StandardScaler |
| `label_encoder.pkl` | Grid ID → class index mapping |
| `evaluation_report.png` | Confusion matrix + feature importance |
| `alert_output.json` | Structured early warning report |
| `alert_map.html` | Interactive Folium prediction map |

---

## 🧠 Feature Set (29 features)

| Layer | Features |
|-------|----------|
| **Movement (A)** | `step_dist_m`, `turning_angle`, `time_delta_hours`, `log_step_dist`, `cos_angle`, `sin_angle` |
| **Satellite (B/D)** | `ndvi`, `landcover_class`, `is_forest`, `is_cropland`, `rainfall_7d_mm` |
| **Human (C)** | `village_distance_m`, `road_density`, `cropland_pct`, `village_risk`, `crop_attraction` |
| **Temporal (D)** | `season`, `time_of_day`, `month`, `hour`, `is_nighttime` |
| **Memory (E)** | `repeat_count`, `seasonal_repeat`, `success_score`, `visit_count`, `last_visit_days_ago`, `is_home_range_core` |

---

## 🚨 Alert Output Example

```json
{
  "timestamp": "2015-08-01T10:00:00",
  "elephant_id": "E1",
  "prediction_horizon_hours": 48,
  "top_prediction": {
    "grid_id": "R0045_C0023",
    "probability": 0.42,
    "coordinates": [25.12, -18.34]
  },
  "at_risk_villages": [
    { "name": "Kasane", "distance_m": 3200.0, "risk_level": "HIGH" }
  ],
  "recommended_action": "IMMEDIATE DISPATCH: Deploy ranger unit to intercept."
}
```

---

## 📋 Requirements

- Python 3.10+
- `pandas`, `numpy`, `geopandas`, `shapely`
- `scikit-learn`, `joblib`
- `earthengine-api`, `geemap` (optional, for satellite features)
- `folium`, `matplotlib`, `seaborn`
- `tqdm`, `scipy`

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Citation

GPS data from:  
> Dryad, doi:10.5061/dryad.dr7sqv9v9 — Elephant movement data, Botswana 2014–2016
