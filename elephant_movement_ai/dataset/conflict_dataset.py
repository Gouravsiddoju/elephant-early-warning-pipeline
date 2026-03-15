import pandas as pd
import os
from typing import List, Dict

class ConflictDataset:
    """Exports village intrusion events."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def save(self, conflicts: List[Dict], filename: str = "conflict_events.csv"):
        """Save a list of conflict observation dicts to CSV."""
        if not conflicts:
            print("No conflicts to save.")
            # Still create an empty CSV to avoid pipeline errors
            df = pd.DataFrame(columns=['timestep', 'village_id', 'distance_to_village', 'herd_size', 'conflict_event'])
        else:
            df = pd.DataFrame(conflicts)
            
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Saved conflict dataset to {filepath} ({len(df)} records)")
