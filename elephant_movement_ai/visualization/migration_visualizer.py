import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from environment.environment import Environment

class MigrationVisualizer:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def plot_migrations(self, env: Environment, datasets_dir: str, filename="migration_arcs.png"):
        print("Rendering Migration Graph...")
        try:
            mig_df = pd.read_csv(f"{datasets_dir}/migration_events.csv")
        except FileNotFoundError:
            return
            
        if mig_df.empty:
            return
            
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(env.base_vegetation, cmap='Greens', alpha=0.2, origin='lower', extent=[-0.5, env.width-0.5, -0.5, env.height-0.5])
        
        # Outline states loosely
        mid_y, mid_x = env.height // 2, env.width // 2
        ax.axhline(mid_y, color='red', linestyle='--', alpha=0.3)
        ax.axvline(mid_x, color='red', linestyle='--', alpha=0.3)
        
        # State centers
        centers = {
            0: (mid_x//2, mid_y//2 + mid_y), # NW
            1: (mid_x + mid_x//2, mid_y//2 + mid_y), # NE
            2: (mid_x//2, mid_y//2), # SW
            3: (mid_x + mid_x//2, mid_y//2) # SE
        }
        
        # Aggregate flow counts
        flow = mig_df.groupby(['from_state', 'to_state']).size().reset_index(name='count')
        
        for idx, row in flow.iterrows():
            if row['from_state'] in centers and row['to_state'] in centers:
                start = centers[row['from_state']]
                end = centers[row['to_state']]
                # Draw arc
                ax.annotate("",
                            xy=end, xycoords='data',
                            xytext=start, textcoords='data',
                            arrowprops=dict(arrowstyle="->",
                                            color="blue",
                                            linewidth=min(10, max(1, row['count']*0.5)),
                                            connectionstyle="arc3,rad=0.3"))
                                            
        ax.set_title("Cross-State Macro Migration Flow")
        fig.savefig(os.path.join(self.output_dir, filename), bbox_inches='tight')
        plt.close(fig)
