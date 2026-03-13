import pandas as pd
import numpy as np
from datetime import datetime

def compute_memory_features(transitions_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (elephant_id, from_grid, to_grid, month):
    - repeat_count: how many times this exact transition occurred in prior years
    - seasonal_repeat: bool — did this transition happen in same month in a prior year?
    - success_score: normalize repeat_count to [0,1]
    Important: only use PAST data — no data leakage.
    """
    print(f"[{datetime.now().isoformat()}] Computing associative memory features from transitions...")
    
    # Ensure sorted chronologically
    df = transitions_df.sort_values(by=['elephant_id', 'Date_Time']).copy()
    
    # Extract month
    df['month'] = df['Date_Time'].dt.month
    
    # 1. repeat_count: how many times this EXACT transition occurred previously
    df = df.sort_values(['elephant_id', 'Date_Time']).reset_index(drop=True)
    df['transition_id'] = df['from_grid'] + "_" + df['to_grid']
    df['repeat_count'] = df.groupby(['elephant_id', 'transition_id']).cumcount()
    
    # 2. seasonal_repeat: did it happen in the SAME MONTH previously?
    df['seasonal_repeat'] = df.groupby(['elephant_id', 'transition_id', 'month']).cumcount() > 0
    df['seasonal_repeat'] = df['seasonal_repeat'].astype(int)
    
    # 3. success_score: normalize repeat_count to [0,1]
    # We divide by the maximum repeat count seen up to this row for the given elephant
    df['max_repeats_so_far'] = df.groupby('elephant_id')['repeat_count'].cummax()
    df['max_repeats_so_far'] = df['max_repeats_so_far'].replace(0, 1) # avoid div by zero
    df['success_score'] = df['repeat_count'] / df['max_repeats_so_far']
    
    # Drop temp cols
    df = df.drop(columns=['transition_id', 'max_repeats_so_far'])
    
    print(f"[{datetime.now().isoformat()}] Memory features computed.")
    return df

def compute_site_fidelity(gps_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each grid cell per elephant:
    - visit_count: total visits ever (cumulative up to point)
    - last_visit_days_ago: days since last visit
    - is_home_range_core: bool — visited in >30% of all fixes (up to point)
    """
    print(f"[{datetime.now().isoformat()}] Computing site fidelity features...")
    
    # Sort strictly by time to avoid leakage!
    df = gps_df.sort_values(by=['id', 'Date_Time']).copy()
    
    # 1. visit_count (cumulative)
    # cumcount() gives counts prior to the current row in the group
    df['visit_count'] = df.groupby(['id', 'grid_id']).cumcount()
    
    # 2. last_visit_days_ago
    df['prev_visit_time'] = df.groupby(['id', 'grid_id'])['Date_Time'].shift(1)
    df['last_visit_days_ago'] = (df['Date_Time'] - df['prev_visit_time']).dt.total_seconds() / (24 * 3600)
    # Fill NA with a high number (e.g. 9999) indicating first visit
    df['last_visit_days_ago'] = df['last_visit_days_ago'].fillna(9999.0)
    
    # 3. is_home_range_core
    df['total_fixes_so_far'] = df.groupby('id').cumcount() + 1
    df['visit_pct'] = df['visit_count'] / df['total_fixes_so_far']
    df['is_home_range_core'] = (df['visit_pct'] > 0.30).astype(int)
    
    df = df.drop(columns=['prev_visit_time', 'total_fixes_so_far', 'visit_pct'])
    
    print(f"[{datetime.now().isoformat()}] Site fidelity computed.")
    return df

if __name__ == "__main__":
    pass
