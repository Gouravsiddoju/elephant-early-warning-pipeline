
import pandas as pd
import os

base_path = r'c:\PROJECTS\Elephant_tracking\early_warning_pipeline'
centroid_path = os.path.join(base_path, 'grid_centroids.csv')

df = pd.read_csv(centroid_path)
centroid_map = df.set_index('grid_id').to_dict('index')

target_id = "R0105_C0170"
print(f"Target ID: '{target_id}'")
print(f"Exists in map: {target_id in centroid_map}")

# Check key types and any matches
keys = list(centroid_map.keys())
print(f"First 5 keys: {keys[:5]} (types: {[type(k) for k in keys[:5]]})")

matches = [k for k in keys if "R0105_C0170" in str(k)]
print(f"Matches for 'R0105_C0170': {matches}")

if matches:
    print(f"Match 0 details: {centroid_map[matches[0]]}")
