from typing import Tuple, List
from .memory_model import MemoryModel

class ElephantHerd:
    """Represents a single elephant herd agent."""
    
    def __init__(self, herd_id: int, start_x: int, start_y: int, herd_size: int, env_width: int, env_height: int):
        self.herd_id = herd_id
        self.leader_position = (start_x, start_y)
        self.herd_size = herd_size
        self.energy_level = 100.0
        
        # Each herd has its own spatial memory of the environment
        self.memory_model = MemoryModel(env_width, env_height)
        
        # Track history for dataset generation
        self.trajectory_history: List[Tuple[int, int]] = [self.leader_position]
        
        # Current cardinal/ordinal direction (e.g., "N", "SE")
        self.current_direction: str = "WAIT"

    def update_position(self, new_x: int, new_y: int, direction_str: str):
        """Update the leader's position and track history."""
        self.leader_position = (new_x, new_y)
        self.current_direction = direction_str
        self.trajectory_history.append(self.leader_position)

    def learn_from_environment(self, crop_density: float, water_presence: float):
        """Update memory based on the resources at the current position."""
        # Reward is higher for water and crops
        reward = (crop_density * 2.0) + (water_presence * 5.0)
        
        if reward > 0:
            x, y = self.leader_position
            self.memory_model.update_memory(x, y, reward)
