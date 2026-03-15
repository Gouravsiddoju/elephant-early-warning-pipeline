import numpy as np

class WeatherEngine:
    """Simulates dynamic weather patterns modifying the ecological landscape."""
    
    def __init__(self, config: dict):
        weather_cfg = config.get('weather', {})
        self.base_temp = weather_cfg.get('base_temperature', 25.0)
        self.temp_var = weather_cfg.get('temp_variance', 10.0)
        self.base_rain = weather_cfg.get('base_rainfall', 5.0)
        self.drought_chance = weather_cfg.get('drought_chance', 0.1)
        self.storm_chance = weather_cfg.get('storm_chance', 0.05)
        
        self.rng = np.random.default_rng(config['simulation'].get('random_seed', 42))
        
        # State
        self.current_temperature = self.base_temp
        self.current_rainfall = self.base_rain
        self.drought_index = 0.0  # 0.0 (no drought) to 1.0 (severe drought)
        self.is_storming = False
        self.day_counter = 0

    def step_day(self):
        """Advances weather by one day, evolving patterns procedurally."""
        self.day_counter += 1
        
        # Temperature fluctuates around base
        self.current_temperature = self.base_temp + self.rng.uniform(-self.temp_var, self.temp_var)
        
        # Rainfall
        if self.rng.random() < self.storm_chance:
            self.current_rainfall = self.base_rain * self.rng.uniform(3.0, 5.0)
            self.is_storming = True
        else:
            self.current_rainfall = self.base_rain * self.rng.uniform(0.1, 1.5)
            self.is_storming = False
            
        # Drought mechanics
        if self.current_rainfall < (self.base_rain * 0.5):
            # Dry day, drought increases
            self.drought_index = min(1.0, self.drought_index + 0.05)
        elif self.current_rainfall > (self.base_rain * 1.5):
            # Wet day, drought resets quickly
            self.drought_index = max(0.0, self.drought_index - 0.2)
            
    def get_ecological_modifiers(self) -> dict:
        """Returns modifiers that affect vegetation and water layers."""
        # Drought reduces available water and vegetation density
        water_mult = max(0.1, 1.0 - (self.drought_index * 0.8))  # Can drop to 20% normal
        veg_mult = max(0.3, 1.0 - (self.drought_index * 0.5))    # Can drop to 50% normal
        
        # Heat makes moving in the day extremely costly
        heat_stress = max(0.0, (self.current_temperature - 30.0) / 10.0)
        
        return {
            'water_multiplier': water_mult,
            'vegetation_multiplier': veg_mult,
            'heat_stress_penalty': heat_stress,
            'is_storming': self.is_storming,
            'drought_index': self.drought_index
        }
