import pandas as pd
import numpy as np

class FeatureEngineering:
    """Joins discrete telemetry and climate data into an ML-ready predictive array."""
    
    def __init__(self):
        # Time window: predict IF a conflict happens in the next N steps
        self.future_window = 144 # 1 day at 10m steps is 144
        
    def build_features(self, datasets_dir: str):
        print("Engineering features from simulation sets...")
        
        # Load bases
        try:
            traj_df = pd.read_csv(f"{datasets_dir}/elephant_trajectories.csv")
            weather_df = pd.read_csv(f"{datasets_dir}/weather_logs.csv")
            conflict_df = pd.read_csv(f"{datasets_dir}/conflict_events.csv")
            crop_df = pd.read_csv(f"{datasets_dir}/crop_damage.csv")
        except Exception as e:
            print("Failed to load datasets. Were they generated?", e)
            return pd.DataFrame(), pd.Series()
            
        if traj_df.empty:
            return pd.DataFrame(), pd.Series()

        # Build massive join index. 
        # Standardize weather to step resolution (forward fill)
        if not weather_df.empty:
            weather_df['day_idx'] = weather_df['timestep'] // 144
            traj_df['day_idx'] = traj_df['timestep'] // 144
            joined = pd.merge(traj_df, weather_df, on='day_idx', how='left')
            joined = joined.ffill()
            joined = joined.fillna(0)
        else:
            joined = traj_df
            joined['temperature'] = 25.0
            joined['rainfall'] = 5.0
            joined['drought_index'] = 0.0

        # Construct target vector: Will a conflict happen near this herd in the next 24 hours?
        if not conflict_df.empty:
            conflict_df['future_conflict'] = 1
        else:
            conflict_df = pd.DataFrame(columns=['timestep', 'herd_id', 'future_conflict'])
            
        # Add crops as conflict type
        if not crop_df.empty:
            crop_df['future_conflict'] = 1
            # Simple union of all bad events
            all_conflicts = pd.concat([
                conflict_df[['timestep', 'future_conflict']], 
                crop_df[['timestep', 'future_conflict']]
            ]).drop_duplicates()
        else:
            all_conflicts = conflict_df
            
        # Target assignment (Binary array)
        y = np.zeros(len(joined))
        
        # Handle overlapping column names from merge
        if 'timestep_x' in joined.columns:
            joined.rename(columns={'timestep_x': 'timestep'}, inplace=True)
            
        # A naive rolling window merge for targets
        if not all_conflicts.empty: # Sort arrays to efficiently search windows
            conflict_times = all_conflicts['timestep'].values
            for i, ts in enumerate(joined['timestep'].values):
                # Check if any conflict event happens between ts and ts + 144
                if len(conflict_times[(conflict_times >= ts) & (conflict_times <= ts + self.future_window)]) > 0:
                    y[i] = 1

        # Drop useless telemetry IDs for Random Forest
        X = joined.drop(columns=['elephant_id', 'herd_id', 'day_idx', 'timestep_y'], errors='ignore')
        if 'timestep_x' in X.columns:
            X.rename(columns={'timestep_x': 'timestep'}, inplace=True)
            
        return X, pd.Series(y, name='target')
