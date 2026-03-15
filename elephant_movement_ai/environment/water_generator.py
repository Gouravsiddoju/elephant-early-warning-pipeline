import numpy as np

class WaterGenerator:
    """Generates rivers and static water bodies (ponds/lakes)."""
    
    def __init__(self, width: int = 100, height: int = 100, seed: int = 42):
        self.width = width
        self.height = height
        self.rng = np.random.default_rng(seed + 2)
        
    def generate_water(self, num_bodies: int = 5, num_rivers: int = 2) -> np.ndarray:
        """
        Generate a water presence matrix (0.0 or 1.0).
        Includes static water bodies and river lines.
        """
        water_layer = np.zeros((self.height, self.width))
        
        # 1. Ponds / Lakes
        for _ in range(num_bodies):
            cx = self.rng.integers(0, self.width)
            cy = self.rng.integers(0, self.height)
            radius = self.rng.integers(2, 6)
            
            y, x = np.ogrid[-cy:self.height-cy, -cx:self.width-cx]
            mask = x**2 + y**2 <= radius**2
            water_layer[mask] = 1.0
            
        # 2. Rivers (Random walks across the grid)
        for _ in range(num_rivers):
            # Start on left or top edge
            if self.rng.random() > 0.5:
                # Left edge
                cx, cy = 0, self.rng.integers(0, self.height)
            else:
                # Top edge
                cx, cy = self.rng.integers(0, self.width), 0
                
            thickness = self.rng.integers(1, 3)
            
            while 0 <= cx < self.width and 0 <= cy < self.height:
                # Mark current pos
                y, x = np.ogrid[-cy:self.height-cy, -cx:self.width-cx]
                mask = x**2 + y**2 <= thickness**2
                water_layer[mask] = 1.0
                
                # Move
                dx = self.rng.choice([0, 1, 1, 2]) # Bias right
                dy = self.rng.choice([-1, 0, 1, 2]) # Bias down
                cx += dx
                cy += dy
                
        return np.clip(water_layer, 0.0, 1.0)
