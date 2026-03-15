class RewardSystem:
    """Calculates reinforcement learning rewards based on local conditions and agent state."""
    
    def __init__(self, config: dict):
        self.weights = config['movement'].get('rewards', {})
        self.r_food = self.weights.get('r_food', 3.0)
        self.r_water = self.weights.get('r_water', 2.0)
        self.r_forest = self.weights.get('r_forest_safety', 1.0)
        self.r_human = self.weights.get('r_human_dist', -2.0)
        self.r_slope = self.weights.get('r_slope_penalty', -1.0)
        self.r_crop = self.weights.get('r_crop_feeding', 5.0)
        self.r_conflict = self.weights.get('r_conflict_event', -10.0)

    def calculate_step_reward(self, features: dict, action: int, is_solitary: bool = False, x: int = 0, y: int = 0, env_width: int = 100, env_height: int = 100) -> float:
        """
        Calculates the instantaneous reward for landing on a cell.
        Solitary elephants value crops higher than herd-bound elephants.
        """
        reward = 0.0
        
        # Boundary penalty to prevent corner hugging
        margin = 10
        if x < margin or x > env_width - margin or y < margin or y > env_height - margin:
            reward -= 5.0 # Large penalty for being near the edge
        
        # Base ecological resources
        if features['vegetation_density'] > 0.4:
            reward += self.r_food * features['vegetation_density']
            reward += self.r_forest  # Safety parameter
            
        if features['water_presence'] > 0:
            reward += self.r_water
            
        # Penalties
        reward += self.r_human * features['human_density']
        reward += self.r_slope * features['terrain_slope']
        reward -= features['heat_stress'] * 2.0  # Big penalty for moving in high heat
        
        # Crop feeding
        if features['crop_density'] > 0.0:
            crop_bonus = self.r_crop * features['crop_density']
            if is_solitary:
                crop_bonus *= 1.5  # Solitary males take higher risks for high rewards
            reward += crop_bonus
            
        return reward
        
    def calculate_conflict_penalty(self) -> float:
        """Triggered when an event generator explicitly registers a bad interaction."""
        return self.r_conflict
