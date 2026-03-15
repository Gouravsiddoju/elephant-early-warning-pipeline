import pandas as pd
import os
from typing import List, Dict

class MigrationDataset:
    """Exports cross-state transitions signaling drought/scarcity migrations."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.filepath = os.path.join(output_dir, "migration_events.csv")
        
    def save(self, observations: List[Dict]):
        if not observations:
            pd.DataFrame(columns=['timestep', 'herd_id', 'from_state', 'to_state']).to_csv(self.filepath, index=False)
            return
            
        df = pd.DataFrame(observations)
        df.to_csv(self.filepath, index=False)
        print(f"Saved migration data to {self.filepath}")
