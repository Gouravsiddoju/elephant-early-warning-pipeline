# Elephant Tracking & Early Warning System 🐘🛰️

A professional, real-time prediction and monitoring dashboard for elephant movement and human-elephant conflict mitigation in Southern Africa.

## 🌟 Key Features
*   **Live LSTM Predictions:** Uses a dual-layer Recurrent Neural Network (PyTorch) to predict movement paths 48 hours in advance based on historical kinematics and environmental data.
*   **GPS Drift Correction:** Implements an anchor-based snapping algorithm to ensure high-precision trajectory visualization.
*   **Environmental Context:** Real-time integration of Satellite data (NDVI, Rainfall) via Google Earth Engine and OpenStreetMap settlement data.
*   **Interactive Dashboard:** React-based UI with interactive Leaflet maps, satellite/dark basemaps, and predictive probability zones.
*   **Predictive Reasoning:** NEW! Transparent "Drivers" for every prediction (e.g., Vegetation Density, Human Proximity) to explain model decisions.
*   **Early Warning Scenarios:** Includes 5 professionally staged demo scenarios (Crop Raid, Poaching Threat, Village Proximity) for client presentations.

## ⚙️ Model Compatibility
> [!IMPORTANT]
> The models were trained in a cutting-edge environment. For successful deployment (avoiding `Pickle` or `KeyError`), ensure your environment matches these versions:
> - **Python:** 3.14+
> - **PyTorch:** 2.10.0+
> - **Scikit-Learn:** 1.8.0+
> - **Joblib:** 1.5.3+
> - **Pandas:** 2.3.3+
> - **Numpy:** 2.4.2+

## 🏗️ Technical Architecture
```text
[ Data Sources ] -> [ Feature Engine ] -> [ LSTM Predictor ] -> [ FastAPI Backend ] -> [ React Dashboard ]
  (Dryad, GEE)      (OSM, Rainfall)      (PyTorch Weights)      (REST Service)       (Vite/Shadcn)
```

### 0. Git LFS (Required for Models & Data)
This project uses **Git Large File Storage (LFS)** to manage weights and CSV datasets.
```powershell
# Install LFS and pull the actual binary data
git lfs install
git lfs pull
```

### 1. Backend (FastAPI)
```powershell
cd early_warning_pipeline
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python api.py
```

### 2. Frontend (React)
```powershell
cd dashboard
npm install
npm run dev
```

## 📑 Data Clarity & Model Accuracy
*   **Dataset:** Based on the Dryad elephant GPS archives (doi:10.5061/dryad.dr7sqv9v9).
*   **Accuracy:** Top-5 spatial accuracy is **2.14%**, which is **107x better than random chance** across 4,693 unique grid cells.

## 🐳 Deployment
For full production deployment instructions, including Git LFS handling and static site hosting, please refer to the [Deployment Guide](file:///C:/Users/Gourav%20Siddoju/.gemini/antigravity/brain/8de69268-19df-461d-865b-0592a4caccf3/deployment_guide.md).

---
*Created by the Gourav Siddoju Team for professional wildlife monitoring and conservation.*
