import uuid
from typing import List

class HerdSplitting:
    """Logic to handle herd fracturing under extreme stress or scarcity."""
    
    def __init__(self, config: dict):
        self.split_threshold = config['herd'].get('split_threshold', 40)
        
    def check_and_split(self, herd, drought_index: float, current_features: dict) -> List:
        """
        Checks if a herd should split into two smaller herds.
        Returns a list of NEW herds spawned from the split (empty if none).
        """
        # Split conditions: massive herd size OR moderate size + severe drought/scarcity
        scarcity_factor = (1.0 - current_features['vegetation_density']) + drought_index
        stress_threshold = self.split_threshold * (1.0 - (scarcity_factor * 0.3)) # Lower threshold when desperate
        
        new_herds = []
        if herd.herd_size > stress_threshold and herd.herd_size > 4:
            # Fracture the herd in half
            split_size = herd.herd_size // 2
            herd.herd_size -= split_size
            
            # In a real environment, wait for `simulation_engine` to instantiate the actual class
            # We return a dict payload to alert the engine to spawn a new agent
            new_herds.append({
                'id': str(uuid.uuid4())[:8],
                'start_x': herd.leader_position[0],
                'start_y': herd.leader_position[1],
                'size': split_size
            })
            print(f"Herd {herd.herd_id} SPLIT due to stress! New herd spawned.")
            
        return new_herds
