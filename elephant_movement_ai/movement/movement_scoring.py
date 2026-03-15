class MovementScoring:
    """Computes the attractiveness score of a given cell based on ecological drivers."""
    
    def __init__(self, config: dict):
        # Extract base weights
        weights = config['movement'].get('weights', {})
        self.w_veg = weights.get('w1_vegetation', 0.3)
        self.w_crop = weights.get('w2_crop', 0.4)
        self.w_water = weights.get('w3_water', 0.25)
        self.w_corridor = weights.get('w4_corridor', 0.2)
        self.w_human = weights.get('w5_human', 0.35)
        self.w_slope = weights.get('w6_slope', 0.1)
        self.w_memory = weights.get('w7_memory', 0.2)
        
        self.temporal_modifiers = config['movement'].get('temporal_modifiers', {})

    def _get_temporal_multipliers(self, time_of_day: str):
        """Retrieve multipliers for the current time phase (day, evening, night, morning)."""
        mods = self.temporal_modifiers.get(time_of_day, {})
        return {
            'veg_mult': mods.get('vegetation_mult', 1.0),
            'crop_mult': mods.get('crop_mult', 1.0),
            'water_mult': mods.get('water_mult', 1.0)
        }

    def compute_score(self, features: dict, memory_value: float, time_of_day: str) -> float:
        """
        Calculate the attractiveness score for a single cell.
        """
        mods = self._get_temporal_multipliers(time_of_day)
        
        score = (
            + (self.w_veg * features['vegetation_density'] * mods['veg_mult'])
            + (self.w_crop * features['crop_density'] * mods['crop_mult'])
            + (self.w_water * features['water_presence'] * mods['water_mult'])
            + (self.w_corridor * features['corridor_probability'])
            - (self.w_human * features['human_density'])
            - (self.w_slope * features['terrain_slope'])
            + (self.w_memory * memory_value)
        )
        return score
