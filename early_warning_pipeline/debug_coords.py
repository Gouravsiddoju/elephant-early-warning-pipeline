import pandas as pd
import pyproj
from grid_builder import build_grid

df = pd.read_csv('feature_matrix.csv')
class_counts = df['to_grid'].value_counts()
valid = class_counts[class_counts >= 10].index
df = df[df['to_grid'].isin(valid)].copy()

proj = pyproj.Transformer.from_crs('EPSG:4326','EPSG:32734',always_xy=True)
min_x, min_y = proj.transform(23.0, -22.0)
max_x, max_y = proj.transform(28.0, -17.0)
grid_gdf = build_grid((min_x, min_y, max_x, max_y), cell_size_m=5000)
grid_wgs84 = grid_gdf.to_crs('EPSG:4326').set_index('grid_id')

centroids = pd.read_csv('grid_centroids.csv')
centroid_map = centroids.set_index('grid_id').to_dict('index')

counts = df.groupby('elephant_id').size().sort_values(ascending=False)
selected = counts[counts >= 10].index[:8].tolist()

print(f"Selected elephants: {selected}")
print()
for eid in selected:
    grp = df[df['elephant_id'] == eid].sort_values('Date_Time')
    cur_grid = grp['from_grid'].iloc[-1]
    in_wgs = cur_grid in grid_wgs84.index
    in_cent = cur_grid in centroid_map
    status = "OK" if (in_wgs or in_cent) else "MISSING!!"
    print(f"Elephant {eid}: grid={cur_grid}, in_grid={in_wgs}, in_centroids={in_cent} [{status}]")
    if not (in_wgs or in_cent):
        sample = grp['from_grid'].unique()[:5].tolist()
        in_any = [g for g in sample if g in grid_wgs84.index or g in centroid_map]
        print(f"  Sample grids: {sample}")
        print(f"  Resolvable grids: {in_any}")
