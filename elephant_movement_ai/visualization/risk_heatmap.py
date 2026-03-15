import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from matplotlib.patches import Circle
from environment.environment import Environment
from ml.risk_model import RiskModel

class RiskHeatmap:
    """Visualizes the ML predicted risk over villages."""
    
    def __init__(self, config: dict):
        self.output_dir = config['paths'].get('visualizations_dir', 'output/visualizations/')
        os.makedirs(self.output_dir, exist_ok=True)
        
    def plot_risk(self, env: Environment, model: RiskModel, features_df: pd.DataFrame, save_filename="risk_heatmap.png"):
        print("Rendering Risk Heatmap...")
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Base map (dimmed)
        ax.imshow(env.base_vegetation, cmap='Greys', alpha=0.2, origin='lower', extent=[-0.5, env.width-0.5, -0.5, env.height-0.5])
        
        if not features_df.empty and model.is_trained:
            # Drop target explicitly if passed
            X = features_df.copy()
            if 'target' in X.columns:
                X = X.drop(columns=['target'])
                
            risk_probs = model.predict_risk(X)
            X['predicted_risk'] = risk_probs
            
            # Map risk back to coordinates (approximate via location bins if necessary)
            # Since features are step-by-step telemetry, simply scattering X/Y risk is fine
            if 'x' in X.columns and 'y' in X.columns:
                 # Subsample for render speed
                 if len(X) > 2000:
                    sample = X.sample(n=2000, random_state=42)
                 else:
                    sample = X
                 scatter = ax.scatter(sample['x'], sample['y'], c=sample['predicted_risk'], cmap='Reds', s=10, alpha=0.5)
                 plt.colorbar(scatter, label='Conflict Probability (Next 24 Hrs)')
                 
        # Overlay Villages
        vx = [v['x'] for v in env.villages]
        vy = [v['y'] for v in env.villages]
        ax.scatter(vx, vy, c='blue', marker='v', s=80, edgecolors='black', label='Villages')
        
        ax.set_title("ML Predicted Human-Elephant Conflict Zones")
        ax.legend()
        fig.savefig(os.path.join(self.output_dir, save_filename), bbox_inches='tight')
        plt.close(fig)
