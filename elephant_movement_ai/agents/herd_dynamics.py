import numpy as np
from typing import List, Tuple

class HerdDynamics:
    """Manages the spatial arrangement of followers and regrouping behavior."""
    
    def __init__(self, config: dict):
        self.rng = np.random.default_rng(config['simulation'].get('random_seed', 42))
        self.herd_radius = config['herd'].get('herd_radius', 5)
        
    def calculate_follower_positions(self, leader_x: int, leader_y: int, herd_size: int, env_width: int, env_height: int) -> List[Tuple[int, int]]:
        """
        Calculates positions for followers such that they remain within 'herd_radius'.
        Followers implicitly 'regroup' if they drift too far out organically.
        """
        positions = []
        num_followers = max(0, herd_size - 1)
        
        for _ in range(num_followers):
            # Followers drift slightly but stay bound by the radius
            dx = int(self.rng.normal(0, self.herd_radius / 2))
            dy = int(self.rng.normal(0, self.herd_radius / 2))
            
            # Clip rigidly to radius if normal dist blows out
            if dx**2 + dy**2 > self.herd_radius**2:
                # normalize to edge of circle
                angle = np.arctan2(dy, dx)
                dx = int(np.cos(angle) * self.herd_radius)
                dy = int(np.sin(angle) * self.herd_radius)
                
            fx = max(0, min(env_width - 1, leader_x + dx))
            fy = max(0, min(env_height - 1, leader_y + dy))
            positions.append((fx, fy))
                
        return positions
