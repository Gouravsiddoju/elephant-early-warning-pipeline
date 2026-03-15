import pandas as pd
import os
from typing import List, Dict

class WeatherDataset:
    """Exports daily weather logs."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.filepath = os.path.join(output_dir, "weather_logs.csv")
        
    def save(self, observations: List[Dict]):
        if not observations:
            pd.DataFrame(columns=['timestep', 'temperature', 'rainfall', 'drought_index']).to_csv(self.filepath, index=False)
            return
            
        df = pd.DataFrame(observations)
        df.to_csv(self.filepath, index=False)
        print(f"Saved weather data to {self.filepath}")
