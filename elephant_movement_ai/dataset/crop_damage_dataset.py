import pandas as pd
import os
from typing import List, Dict

class CropDamageDataset:
    """Exports explicit crop feeding records linked to village ids."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.filepath = os.path.join(output_dir, "crop_damage.csv")
        
    def save(self, observations: List[Dict]):
        if not observations:
            pd.DataFrame(columns=['timestep', 'village', 'crop_damage', 'herd_size']).to_csv(self.filepath, index=False)
            return
            
        df = pd.DataFrame(observations)
        df.to_csv(self.filepath, index=False)
        print(f"Saved crop damage data to {self.filepath}")
