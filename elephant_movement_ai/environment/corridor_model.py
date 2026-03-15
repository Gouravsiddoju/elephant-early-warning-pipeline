import numpy as np
from scipy.ndimage import gaussian_filter

class CorridorModel:
    """Generates ecological corridors, increasing movement probability between forests."""
    
    def __init__(self, width: int = 100, height: int = 100, seed: int = 42):
        self.width = width
        self.height = height
        self.rng = np.random.default_rng(seed + 4)
        
    def generate_corridors(self, veg_layer: np.ndarray, num_corridors: int = 3) -> np.ndarray:
        """
        Creates probabilistic corridors connecting dense vegetation zones.
        """
        corridor_layer = np.zeros((self.height, self.width))
        
        # Simple heuristic: pick random high-veg points and draw lines between them, then smooth
        high_veg_indices = np.argwhere(veg_layer > 0.6)
        
        if len(high_veg_indices) < 2:
            return corridor_layer # Not enough forest to link
            
        for _ in range(num_corridors):
            idx1 = self.rng.integers(0, len(high_veg_indices))
            idx2 = self.rng.integers(0, len(high_veg_indices))
            
            y1, x1 = high_veg_indices[idx1]
            y2, x2 = high_veg_indices[idx2]
            
            # Draw line between them (Bresenham-style approx using linspace)
            num_points = int(np.hypot(x2 - x1, y2 - y1) * 2)
            if num_points == 0:
                continue
                
            x_vals = np.linspace(x1, x2, num_points).astype(int)
            y_vals = np.linspace(y1, y2, num_points).astype(int)
            
            # Clip
            x_vals = np.clip(x_vals, 0, self.width - 1)
            y_vals = np.clip(y_vals, 0, self.height - 1)
            
            corridor_layer[y_vals, x_vals] = 1.0
            
        # Smooth the lines out heavily so they act as broad zones of probability
        smoothed = gaussian_filter(corridor_layer, sigma=4.0)
        if smoothed.max() > 0:
            smoothed = smoothed / smoothed.max()
            
        return smoothed
