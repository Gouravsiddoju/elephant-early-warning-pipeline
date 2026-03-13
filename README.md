# 🐘 Elephant Movement Prediction & Early Warning Pipeline

A production-quality AI pipeline that learns from historical elephant GPS data to **predict herd movement** and **generate early warnings** for at-risk villages — helping prevent human-elephant conflict in Botswana.

---

## 🏆 Model Performance (LSTM v2 — Current)

| Metric | Value |
|--------|-------|
| Top-1 Grid Accuracy | **77.6%** |
| Mean spatial error | **0.53 km** |
| Median spatial error | **0.06 km (60 metres)** |
| Predictions within 5 km | **99.9%** |
| Predictions within 10 km | **99.96%** |

> Trained on 166K sequences · 37 epochs · NVIDIA T600 GPU · ~10 min

---

## 🗺️ Overview

```
Real GPS Data (Dryad) + OSM Botswana + Google Earth Engine
              ↓
    Spatial Grid (5km cells, UTM 34S)
              ↓
  Feature Engineering (5 Dataset Layers, 28 features)
              ↓
   ElephantLSTMv2 — Coordinate Regression
   (predicts Δlat, Δlon → snaps to nearest grid cell)
              ↓
  Top-5 Grid Cell Predictions with spatial probabilities
              ↓
   Village Risk Alert + Interactive Folium Map
```

**Study Region:** Botswana (~23–28°E, -22 to -17°S)  
**Period:** April 2014 – December 2016 (Dryad dataset)  
**Individuals:** 11 GPS-tracked elephants · 213K fixes

---

## 📁 Pipeline Architecture

```
Stage 1   data_loader.py       → Load & validate Dryad CSV
Stage 2   grid_builder.py      → Build 5km UTM spatial grid, compute transitions
Stage 3   gee_extractor.py     → NDVI, rainfall, land cover via Google Earth Engine
Stage 4   human_features.py    → Village distances, road density, cropland (OSM)
Stage 5   memory_features.py   → Repeat transitions, site fidelity (no data leakage)
Stage 6   feature_matrix.py    → Merge all 5 layers, engineer 28 features
Stage 7   model_trainer.py     → LSTM v1 classifier (legacy, kept as fallback)
          improved_trainer.py  → LSTM v2 regression trainer (recommended)
Stage 8   predictor.py         → Unified predictor: auto-detects v1/v2 model
Stage 9   alert_engine.py      → 5km village buffer, risk levels, Folium map
Stage 10  main.py              → End-to-end orchestrator
```

---

## 🧠 Model Architecture — ElephantLSTMv2

The v2 model replaces the 4,693-class classifier with **coordinate regression**, which is far more spatially accurate.

```
Input: sequence of 10 GPS fixes × 28 features
         ↓
3-layer LSTM (hidden=256, dropout=0.3)
         ↓
LayerNorm  →  Soft Attention (weighted sum over 10 steps)
         ↓
FC: 256 → 128 → GELU → Dropout → 64 → GELU → 2
         ↓
Output: (Δlat, Δlon) offset from current position
         ↓
Snap to nearest grid cell centroid
```

**Loss:** MSE on (Δlat, Δlon) — numerically stable, no haversine in gradient path  
**Optimizer:** AdamW (lr=5e-4, weight_decay=1e-4)  
**LR schedule:** Cosine Annealing (T_max=200, eta_min=1e-6)  
**Early stopping:** patience=20 epochs

### Why regression beats classification

| Approach | Classes | Top-1 Acc | Avg Error |
|----------|---------|-----------|-----------|
| Random Forest (original) | 4,693 | ~2–5% | ~30–50 km |
| LSTM Classifier (v1) | 4,693 | 0.6% | ~30 km |
| **LSTM Regression (v2)** | **—** | **77.6%** | **0.53 km** |

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
git clone https://github.com/Gouravsiddoju/elephant-early-warning-pipeline.git
cd elephant-early-warning-pipeline

# 2. Install dependencies (includes PyTorch with CUDA support)
pip install -r requirements.txt

# 3. (Optional) Authenticate Google Earth Engine for satellite features
earthengine authenticate

# 4. Place your data files:
#    - doi_10_5061_dryad_dr7sqv9v9__v20200116/ElephantsData_ano.csv
#    - botswana-260310-free.shp/  (Botswana OSM shapefiles folder)

# 5. Run the full pipeline (builds features + trains + generates alert)
python main.py

# 6. Or train the improved v2 model standalone
python improved_trainer.py

# 7. Run a demo scenario
python demo_scenarios.py --scenario conflict   # CONFLICT demo
python demo_scenarios.py --scenario safe       # SAFE demo
python demo_scenarios.py --scenario test_data  # Real test row from dataset
```

---

## 📦 Output Files

| File | Description |
|------|-------------|
| `feature_matrix.csv` | 28-feature merged training dataset |
| `elephant_lstm_v2.pt` | Trained LSTM v2 model weights (recommended) |
| `elephant_lstm.pt` | Trained LSTM v1 classifier weights (fallback) |
| `scaler_v2.pkl` | Fitted StandardScaler for v2 |
| `feature_names_v2.pkl` | Ordered feature list for v2 |
| `grid_centroids.csv` | Grid cell ID → centroid lat/lon map |
| `label_encoder.pkl` | Grid ID → class index (v1 only) |
| `evaluation_report_v2.png` | Spatial error distribution histogram |
| `training_log_v2.txt` | Full epoch-by-epoch training log |
| `alert_output.json` | Structured early warning report |
| `alert_map.html` | Interactive Folium prediction map |

---

## 🧩 Feature Set (28 features)

| Layer | Features |
|-------|----------|
| **Movement** | `step_dist_m`, `turning_angle`, `time_delta_hours`, `log_step_dist`, `cos_angle`, `sin_angle` |
| **Satellite** | `ndvi`, `landcover_class`, `is_forest`, `is_cropland`, `rainfall_7d_mm` |
| **Human** | `village_distance_m`, `road_density`, `cropland_pct`, `village_risk`, `crop_attraction` |
| **Temporal** | `Season`, `TimeofDay`, `month`, `hour`, `is_nighttime` |
| **Memory** | `repeat_count`, `seasonal_repeat`, `success_score`, `visit_count`, `last_visit_days_ago`, `is_home_range_core` |

---

## 🚨 Alert Output Example

```json
{
  "timestamp": "2016-07-03T16:00:00+00:00",
  "elephant_id": "8",
  "top_prediction": {
    "grid_id": "R0055_C0005",
    "probability": 0.41,
    "centroid_lat": -19.832,
    "centroid_lon": 23.471,
    "dist_km": 0.06
  },
  "at_risk_villages": [
    { "name": "Maun", "distance_m": 3400.0, "risk_level": "HIGH" }
  ],
  "recommended_action": "IMMEDIATE DISPATCH: Deploy ranger unit to intercept."
}
```

---

## 📋 Requirements

- Python 3.10+
- `torch >= 2.1.0` (CUDA recommended: `torch==2.6.0+cu124`)
- `pandas`, `numpy`, `geopandas`, `shapely`, `pyproj`
- `scikit-learn`, `joblib`
- `earthengine-api`, `geemap` (optional — satellite features)
- `folium`, `matplotlib`, `seaborn`
- `tqdm`, `scipy`

Install all with:
```bash
pip install -r requirements.txt
# For GPU (CUDA 12.4):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

---

## 🔧 Key Bug Fixes Applied

| # | File | Fix |
|---|------|-----|
| Grid area | `human_features.py` | Grid area computed from actual geometry (was hardcoded 500m², causing 16× error) |
| Zero padding | `predictor.py` | Short sequences padded with zeros, not first-row copy |
| Double scaling | `demo_scenarios.py` | Raw features passed to predictor (scaler runs only once) |
| CUDA pickle | `model_trainer.py` | Removed `joblib.dump` of CUDA model; only `torch.save(state_dict)` used |
| Timezone | `alert_engine.py` | Fixed IST → CAT (UTC+2, correct for Botswana) |
| Sort order | `memory_features.py` | Explicit sort by `elephant_id + Date_Time` before `cumcount` |

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Citation

GPS data from:  
> Dryad, doi:10.5061/dryad.dr7sqv9v9 — Elephant movement data, Botswana 2014–2016
