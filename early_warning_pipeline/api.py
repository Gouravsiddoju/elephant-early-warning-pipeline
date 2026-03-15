from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import os, json, pandas as pd
from typing import Optional
from prediction_service import PredictionService

app = FastAPI(title="Elephant Tracking Early Warning API")

# Enable CORS for React dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURE_CSV = os.path.join(BASE_DIR, 'feature_matrix.csv')
SCENARIOS_DIR = os.path.join(BASE_DIR, '..', 'dashboard', 'public', 'scenarios')

# Initialize Prediction Service
service = PredictionService(BASE_DIR)

@app.get("/")
def read_root():
    return {"message": "Elephant Tracking API is live", "version": "1.0.0"}

@app.get("/api/dashboard-data")
def get_dashboard_data(scenario_id: Optional[int] = Query(None)):
    """
    Returns dashboard data. If scenario_id is provided, loads the static scenario file.
    Otherwise, runs live inference on feature_matrix.csv.
    """
    if scenario_id:
        scenario_file = os.path.join(SCENARIOS_DIR, f"scenario_{scenario_id}.json")
        if not os.path.exists(scenario_file):
            raise HTTPException(status_code=404, detail=f"Scenario {scenario_id} not found")
        with open(scenario_file, 'r') as f:
            return json.load(f)
    
    # LIVE INFERENCE
    if not os.path.exists(FEATURE_CSV):
        raise HTTPException(status_code=500, detail="Feature matrix not found for live inference")
    
    try:
        df = pd.read_csv(FEATURE_CSV)
        return service.generate_dashboard_data(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/scenarios")
def get_scenarios():
    index_file = os.path.join(SCENARIOS_DIR, "index.json")
    if not os.path.exists(index_file):
        return []
    with open(index_file, 'r') as f:
        return json.load(f)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
