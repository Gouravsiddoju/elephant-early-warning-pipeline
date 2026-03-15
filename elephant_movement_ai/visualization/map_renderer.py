import matplotlib.pyplot as plt
import numpy as np
import os
from environment.environment import Environment

class MapRenderer:
    """Renders base 2D layers of the simulation environment."""
    
    def __init__(self, config: dict):
        self.output_dir = config['paths'].get('visualizations_dir', 'output/visualizations/')
        os.makedirs(self.output_dir, exist_ok=True)
        
    def render_base_map(self, env: Environment, filename="base_map.png"):
        """Plot the core ecological layers."""
        print("Rendering Base Map...")
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 1. Vegetation as heatmap
        im = ax.imshow(env.base_vegetation, cmap='Greens', alpha=0.6, origin='lower', extent=[-0.5, env.width-0.5, -0.5, env.height-0.5])
        
        # 2. Water overlay
        water_pts = np.argwhere(env.base_water > 0.0)
        if len(water_pts) > 0:
            ax.scatter(water_pts[:, 1], water_pts[:, 0], c='blue', s=2, alpha=0.5, label='Water')
            
        # 3. Villages
        vx = [v['x'] for v in env.villages]
        vy = [v['y'] for v in env.villages]
        ax.scatter(vx, vy, c='red', marker='s', s=40, edgecolors='black', label='Villages')
        
        ax.set_title("Elephant Movement - Simulated Environment (200x200)")
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        
        fig.savefig(os.path.join(self.output_dir, filename), bbox_inches='tight')
        plt.close(fig)
        print(f"Base map saved to {self.output_dir}/{filename}")

    def plot_corridors(self, env: Environment, filename="corridors_map.png"):
        print("Rendering Corridors Map...")
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(env.base_vegetation, cmap='Greens', alpha=0.4, origin='lower', extent=[-0.5, env.width-0.5, -0.5, env.height-0.5])
        
        corridor_pts = np.argwhere(env.corridors > 0.1)
        if len(corridor_pts) > 0:
            ax.scatter(corridor_pts[:, 1], corridor_pts[:, 0], c='purple', s=1, alpha=0.3, label='Corridors')
            
        ax.set_title("Generated Ecological Corridors")
        ax.legend()
        fig.savefig(os.path.join(self.output_dir, filename), bbox_inches='tight')
        plt.close(fig)
