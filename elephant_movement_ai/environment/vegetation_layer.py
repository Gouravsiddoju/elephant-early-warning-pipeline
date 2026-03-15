import numpy as np
from scipy.ndimage import gaussian_filter

class VegetationLayer:
    """Generates contiguous forest clusters representing vegetation density."""
    
    def __init__(self, width: int = 100, height: int = 100, seed: int = 42):
        self.width = width
        self.height = height
        self.rng = np.random.default_rng(seed + 1) # offset seed for varied layers
        
    def generate_vegetation(self, smoothing_sigma: float = 6.0, threshold: float = 0.4) -> np.ndarray:
        """
        Generate a vegetation density matrix (0.0 to 1.0).
        Values below 'threshold' are forced to 0 to create distinct sparse areas/grasslands,
        while remaining values represent forests.
        """
        raw_noise = self.rng.random((self.height, self.width))
        smoothed = gaussian_filter(raw_noise, sigma=smoothing_sigma)
        
        # Normalize to 0-1
        normalized = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min())
        
        # Apply threshold to create definitive non-forest regions
        veg_mask = normalized > threshold
        
        # Re-normalize just the forested parts to keep density varied inside boundaries
        result = np.zeros_like(normalized)
        if np.any(veg_mask):
            masked_vals = normalized[veg_mask]
            min_val = masked_vals.min()
            max_val = masked_vals.max()
            result[veg_mask] = (normalized[veg_mask] - min_val) / (max_val - min_val + 1e-9)
            
        return result
