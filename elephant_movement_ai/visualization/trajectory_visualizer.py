import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from environment.environment import Environment

class TrajectoryVisualizer:
    """Overlays herd tracking data onto the environment."""
    
    def __init__(self, config: dict):
        self.output_dir = config['paths'].get('visualizations_dir', 'output/visualizations/')
        os.makedirs(self.output_dir, exist_ok=True)
        
    def plot_trajectories(self, env: Environment, datasets_dir: str, save_filename="rl_herd_trajectories.png"):
        print("Rendering Trajectories map...")
        try:
            df = pd.read_csv(f"{datasets_dir}/elephant_trajectories.csv")
        except FileNotFoundError:
            print("No trajectory dataset found to plot.")
            return

        if df.empty:
            print("Trajectory dataset is empty.")
            return

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(env.base_vegetation, cmap='Greens', alpha=0.3, origin='lower', extent=[-0.5, env.width-0.5, -0.5, env.height-0.5])
        
        for h_id in df['elephant_id'].unique():
            herd_df = df[df['elephant_id'] == h_id]
            # Fast downsample
            if len(herd_df) > 1000:
                herd_df = herd_df.iloc[::(len(herd_df)//1000)]
                
            ax.plot(herd_df['x'], herd_df['y'], marker='.', markersize=2, linestyle='-', linewidth=0.5, alpha=0.8, label=f'{h_id}')
            
            # Start/End
            start = herd_df.iloc[0]
            end = herd_df.iloc[-1]
            ax.scatter(start['x'], start['y'], color='green', marker='^', s=50, zorder=5)
            ax.scatter(end['x'], end['y'], color='red', marker='X', s=50, zorder=5)

        ax.set_title("RL Trained Elephant Pathing")
        
        # Max out legend elements so it doesn't crash plot
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 0:
            ax.legend(handles[:15], labels[:15], loc='upper right', bbox_to_anchor=(1.15, 1))
        
        fig.savefig(os.path.join(self.output_dir, save_filename), bbox_inches='tight')
        plt.close(fig)
        print(f"Trajectories saved to {self.output_dir}/{save_filename}")
