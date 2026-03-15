from typing import List, Dict

class ObservationRecorder:
    """Records trajectories and conflict events sequentially during simulation."""
    
    def __init__(self):
        self.trajectories: List[Dict] = []
        self.conflicts: List[Dict] = []
        
    def record_trajectory(self, timestep: int, herd_id: int, x: int, y: int, size: int):
        self.trajectories.append({
            'timestep': timestep,
            'herd_id': herd_id,
            'x_coordinate': x,
            'y_coordinate': y,
            'herd_size': size
        })
        
    def record_conflict(self, timestep: int, village_id: int, distance: float, size: int):
        self.conflicts.append({
            'timestep': timestep,
            'village_id': village_id,
            'distance_to_village': distance,
            'herd_size': size,
            'conflict_event': 1 # Binary marker
        })
