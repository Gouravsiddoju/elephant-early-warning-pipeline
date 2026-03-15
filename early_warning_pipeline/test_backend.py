
import os
import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
from api import app
from prediction_service import PredictionService

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURE_CSV = os.path.join(BASE_DIR, 'feature_matrix.csv')

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def service():
    return PredictionService(BASE_DIR)

def test_prediction_service_init(service):
    """Verify that the prediction service loads model and grid resources."""
    assert service.model is not None
    assert service.scaler is not None
    assert service.grid_wgs84 is not None
    assert len(service.centroid_map) > 0

def test_trajectory_correction(service):
    """Verify that trajectory anchoring fixes drift."""
    # Create fake history rows
    df = pd.DataFrame({
        'step_dist_m': [1000, 1000, 1000],
        'turning_angle': [0.1, -0.1, 0.2]
    })
    
    anchor_lat, anchor_lon = -19.5, 26.3
    path = service.get_corrected_path(df, anchor_lat, anchor_lon)
    
    # Path should start/end (depending on implementation, here it's 1-shifted)
    # The last point must be exactly the anchor
    assert len(path) == len(df) + 1
    assert path[-1] == [anchor_lat, anchor_lon]
    
    # Check that it's not all zeros
    assert path[0] != [0, 0]

def test_generate_dashboard_data(service):
    """Test dashboard data generation with a sample of the real feature matrix."""
    if not os.path.exists(FEATURE_CSV):
        pytest.skip("feature_matrix.csv not found")
        
    df = pd.read_csv(FEATURE_CSV).head(100)
    data = service.generate_dashboard_data(df, n_elephants=2)
    
    assert "elephants" in data
    assert len(data["elephants"]) > 0
    assert "historyMap" in data
    
    # Check history mapping
    eid = data["elephants"][0]["id"]
    assert eid in data["historyMap"]
    # Verify the last coordinate of the history matches the elephant current position
    hist = data["historyMap"][eid]
    pos = data["elephants"][0]["position"]
    assert hist[-1] == [pos["lat"], pos["lng"]]

def test_api_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "version" in response.json()

def test_api_scenarios(client):
    response = client.get("/api/scenarios")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_api_dashboard_data_live(client):
    """Test the live inference endpoint."""
    response = client.get("/api/dashboard-data")
    # This might take a few seconds as it runs inference
    assert response.status_code == 200
    data = response.json()
    assert "elephants" in data
    assert "generatedAt" in data

def test_api_dashboard_data_scenario(client):
    """Test loading a specific scenario via API."""
    response = client.get("/api/dashboard-data?scenario_id=1")
    assert response.status_code == 200
    data = response.json()
    assert len(data["elephants"]) > 0
