import numpy as np

class VillageGenerator:
    """Generates human settlements and their associated crop fields."""
    
    def __init__(self, width: int = 100, height: int = 100, seed: int = 42):
        self.width = width
        self.height = height
        self.rng = np.random.default_rng(seed + 3)
        self.villages = []
        
    def generate_human_and_crop_layers(self, num_villages: int = 10):
        """
        Generate two layers:
        1. human_density: high near village center
        2. crop_density: distributed around village radius
        Returns (human_density, crop_density, villages_info)
        """
        human_layer = np.zeros((self.height, self.width))
        crop_layer = np.zeros((self.height, self.width))
        self.villages = []
        
        for v_id in range(num_villages):
            cx = self.rng.integers(10, self.width - 10)
            cy = self.rng.integers(10, self.height - 10)
            radius = self.rng.integers(4, 8)
            population = self.rng.integers(50, 500)
            
            self.villages.append({
                'id': v_id,
                'x': cx,
                'y': cy,
                'radius': radius,
                'population': population
            })
            
            y, x = np.ogrid[-cy:self.height-cy, -cx:self.width-cx]
            dist_sq = x**2 + y**2
            
            # Human density (Gaussian decay from center)
            mask_human = dist_sq <= radius**2
            # Max density based on pop
            human_layer[mask_human] += np.exp(-dist_sq[mask_human] / (radius))
            
            # Crop density (Ring-like or patchy around village)
            crop_radius = radius + self.rng.integers(2, 6)
            mask_crop = (dist_sq <= crop_radius**2) & (dist_sq >= (radius//2)**2)
            
            # Patchy crops
            patchy_noise = self.rng.random((len(y), len(x[0]))) # (100, 100) broadcast shape?
            # actually dist_sq is 100x100 so
            patchy_noise = self.rng.random((self.height, self.width))
            
            # apply crops
            valid_crop_area = mask_crop & (patchy_noise > 0.4)
            crop_layer[valid_crop_area] += patchy_noise[valid_crop_area]
            
        human_layer = np.clip(human_layer, 0.0, 1.0)
        crop_layer = np.clip(crop_layer, 0.0, 1.0)
        
        return human_layer, crop_layer, self.villages
