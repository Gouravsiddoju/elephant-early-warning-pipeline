import numpy as np
from scipy.ndimage import gaussian_filter

class TerrainGenerator:
    """Generates the base terrain slope layer using Gaussian smoothed noise."""
    
    def __init__(self, width: int = 100, height: int = 100, seed: int = 42):
        self.width = width
        self.height = height
        self.rng = np.random.default_rng(seed)
        
    def generate_slope_layer(self, smoothing_sigma: float = 5.0) -> np.ndarray:
        """
        Produce a terrain slope layer (0.0 to 1.0).
        Higher values represent steeper terrain which elephants may avoid.
        """
        raw_noise = self.rng.random((self.height, self.width))
        smoothed = gaussian_filter(raw_noise, sigma=smoothing_sigma)
        
        # Normalize to 0-1
        min_val = smoothed.min()
        max_val = smoothed.max()
        normalized = (smoothed - min_val) / (max_val - min_val)
        
        return normalized
