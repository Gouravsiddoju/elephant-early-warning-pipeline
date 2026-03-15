import pandas as pd
import numpy as np
from geopy.distance import geodesic
import warnings
warnings.filterwarnings('ignore')

def analyze_gps_data(csv_file):
    print(f"Loading {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # Clean and sort
    print("Cleaning and sorting...")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["individual-local-identifier", "timestamp"])
    
    # We may have true/True or non-boolean visible column, let's just keep rows with valid coordinates
    df = df.dropna(subset=["location-lat", "location-long"])
    
    # Calculate Distances
    print("Calculating distances and speeds...")
    # Calculate distance grouped by individual to avoid calculating distance between different elephants
    
    def calculate_metrics(group):
        distances = [0]
        for i in range(1, len(group)):
            p1 = (group.iloc[i-1]["location-lat"], group.iloc[i-1]["location-long"])
            p2 = (group.iloc[i]["location-lat"], group.iloc[i]["location-long"])
            distances.append(geodesic(p1, p2).meters)
            
        group["distance"] = distances
        group["time_diff"] = group["timestamp"].diff().dt.total_seconds()
        # Speed in km/h: (meters / seconds) * 3.6
        group["speed_kmh"] = (group["distance"] / group["time_diff"]) * 3.6
        return group
    
    # For speed, let's just subset a bit if it's too large, or we can use vectorized haversine for speed
    # Vectorized haversine is much faster for large datasets:
    def vectorized_haversine(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371000 # Radius of earth in meters
        return c * r
    
    df['prev_lat'] = df.groupby('individual-local-identifier')['location-lat'].shift(1)
    df['prev_lon'] = df.groupby('individual-local-identifier')['location-long'].shift(1)
    
    # Calculate distance using vectorized haversine
    df['distance'] = vectorized_haversine(df['prev_lat'], df['prev_lon'], df['location-lat'], df['location-long'])
    df['distance'] = df['distance'].fillna(0)
    
    df["time_diff"] = df.groupby('individual-local-identifier')["timestamp"].diff().dt.total_seconds()
    
    # Speed in km/h
    df["speed_kmh"] = (df["distance"] / df["time_diff"]) * 3.6
    
    # Clean up infinite or nan speeds
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Statistics
    print("\n--- BEHAVIORAL STATISTICS ---")
    avg_speed = df["speed_kmh"].mean()
    print(f"Average Speed: {avg_speed:.3f} km/h")
    
    max_speed = df["speed_kmh"].max()
    print(f"Max Speed: {max_speed:.3f} km/h")
    
    df["date"] = df["timestamp"].dt.date
    daily_distance = df.groupby(["individual-local-identifier", "date"])["distance"].sum()
    avg_daily_distance = daily_distance.mean() / 1000 # in km
    print(f"Average Daily Distance: {avg_daily_distance:.3f} km")
    
    # Rest vs Movement
    # Threshold in km/h, e.g., 0.3 km/h
    moving_threshold = 0.3 
    df["moving"] = df["speed_kmh"] > moving_threshold
    
    state_percentages = df["moving"].value_counts(normalize=True) * 100
    print("\n--- REST VS MOVEMENT ---")
    print(f"Moving (> {moving_threshold} km/h): {state_percentages.get(True, 0):.2f}%")
    print(f"Resting (<= {moving_threshold} km/h): {state_percentages.get(False, 0):.2f}%")
    
    # Save a sample configuration
    config = {
        "avg_speed": avg_speed,
        "max_speed": max_speed,
        "daily_distance": avg_daily_distance * 1000, # back to meters
        "rest_probability": state_percentages.get(False, 0) / 100.0
    }
    
    print("\n--- ELEPHANT CONFIGURATION FOR SIMULATOR ---")
    print(config)

if __name__ == "__main__":
    dataset_file = "ThermochronTracking Elephants Kruger 2007.csv"
    analyze_gps_data(dataset_file)
