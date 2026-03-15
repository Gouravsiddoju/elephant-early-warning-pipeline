import yaml
import os
import time

from simulation.simulation_engine import SimulationEngine
from dataset.dataset_builder import DatasetBuilder
from ml.training_pipeline import TrainingPipeline
from visualization.map_renderer import MapRenderer
from visualization.trajectory_visualizer import TrajectoryVisualizer
from visualization.risk_heatmap import RiskHeatmap
from visualization.migration_visualizer import MigrationVisualizer

def load_config(config_path="config/simulation_config.yaml") -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def run_full_pipeline():
    print("==================================================")
    print("ELEPHANT MOVEMENT MULTI-AGENT RL & RISK PIPELINE")
    print("==================================================\n")
    
    # 1. Configuration
    try:
        config = load_config()
        print(f"Loaded configuration from config/simulation_config.yaml")
    except Exception as e:
        print(e)
        return
        
    start_time = time.time()
        
    # 2. Simulation Sandbox
    print("\n--- PHASE 1: RL SIMULATION EXECUTING ---")
    sim_engine = SimulationEngine(config)
    
    # Passing pretrain=True will auto-train the PPO agents if a weights file isn't present
    sim_engine.setup(pretrain=True) 
    
    # Execute the learned policies on the environment
    sim_engine.run()
    
    # 3. Datasets
    print("\n--- PHASE 2: DATASET GENERATION ---")
    dataset_builder = DatasetBuilder(config)
    dataset_builder.build_datasets(sim_engine)
    
    # 4. Machine Learning Risk Model
    print("\n--- PHASE 3: ML RISK PREDICTION ---")
    ml_pipeline = TrainingPipeline(config)
    risk_model = ml_pipeline.run()
    
    # 5. Visualizer Outputs
    print("\n--- PHASE 4: VISUALIZATION ARTIFACTS ---")
    # Base Map & Corridors
    map_renderer = MapRenderer(config)
    map_renderer.render_base_map(sim_engine.env)
    map_renderer.plot_corridors(sim_engine.env)
    
    # RL Trajectories
    traj_viz = TrajectoryVisualizer(config)
    traj_viz.plot_trajectories(sim_engine.env, ml_pipeline.datasets_dir)
    
    # Macro Migration Flows
    mig_viz = MigrationVisualizer(config['paths'].get('visualizations_dir', 'output/visualizations/'))
    mig_viz.plot_migrations(sim_engine.env, ml_pipeline.datasets_dir)
    
    # Heatmap
    if risk_model is not None and risk_model.is_trained:
        heatmap_viz = RiskHeatmap(config)
        # Use the raw datasets dataframe explicitly built during model extraction for predictions
        X, _ = ml_pipeline.fe.build_features(ml_pipeline.datasets_dir)
        heatmap_viz.plot_risk(sim_engine.env, risk_model, X)
        
    print(f"\n==================================================")
    print(f"PIPELINE COMPLETE - Total Time: {(time.time() - start_time):.2f}s")
    print("View output images in output/visualizations/ and CSVs in output/datasets/")
    print("==================================================")

if __name__ == "__main__":
    run_full_pipeline()
