import pandas as pd
import os
from typing import List, Dict

class ConflictEventsDataset:
    """Exports destructive/encounter events between humans and elephants."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.filepath = os.path.join(output_dir, "conflict_events.csv")
        
    def save(self, observations: List[Dict]):
        if not observations:
            pd.DataFrame(columns=['timestep', 'village_id', 'event_type', 'herd_size', 'damage_score']).to_csv(self.filepath, index=False)
            return
            
        df = pd.DataFrame(observations)
        df.to_csv(self.filepath, index=False)
        print(f"Saved conflict events data to {self.filepath}")
