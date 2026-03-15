import pandas as pd
import os
from typing import List, Dict

class TrajectoryDataset:
    """Exports raw step-by-step positions for all tracking."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.filepath = os.path.join(output_dir, "elephant_trajectories.csv")
        
    def save(self, observations: List[Dict]):
        if not observations:
            pd.DataFrame(columns=['timestep', 'elephant_id', 'herd_id', 'x', 'y', 'state']).to_csv(self.filepath, index=False)
            return
            
        df = pd.DataFrame(observations)
        df.to_csv(self.filepath, index=False)
        print(f"Saved trajectory data to {self.filepath}")
