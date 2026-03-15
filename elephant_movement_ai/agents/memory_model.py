import numpy as np

class MemoryModel:
    """Represents a herd's spatial memory of high-value resources."""
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        # Initialize memory as 0 everywhere
        self.memory_map = np.zeros((height, width))
        
    def update_memory(self, x: int, y: int, reward: float):
        """
        Updates memory at the current location based on reward.
        Reward is typically determined by crop density and water presence.
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            # Add reward, cap at some maximum memory value (e.g., 10.0)
            self.memory_map[y, x] = min(10.0, self.memory_map[y, x] + reward)
            
    def get_memory_value(self, x: int, y: int) -> float:
        """Returns the memory value at the given location."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.memory_map[y, x]
        return 0.0

    def decay_memory(self, decay_rate: float = 0.01):
        """Optional: slowly decay memory over time to simulate forgetting."""
        self.memory_map = np.maximum(0.0, self.memory_map - decay_rate)
