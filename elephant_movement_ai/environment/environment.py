import numpy as np
from .terrain_generator import TerrainGenerator
from .vegetation_layer import VegetationLayer
from .water_generator import WaterGenerator
from .village_generator import VillageGenerator
from .corridor_model import CorridorModel
from .weather_engine import WeatherEngine

class Environment:
    """Master environment aggregator holding all spatial grids."""
    
    def __init__(self, config: dict):
        self.config = config
        self.width = config['environment'].get('grid_size_x', 200)
        self.height = config['environment'].get('grid_size_y', 200)
        self.seed = config['simulation'].get('random_seed', 42)
        
        self.terrain_gen = TerrainGenerator(self.width, self.height, self.seed)
        self.veg_gen = VegetationLayer(self.width, self.height, self.seed)
        self.water_gen = WaterGenerator(self.width, self.height, self.seed)
        self.village_gen = VillageGenerator(self.width, self.height, self.seed)
        self.corridor_gen = CorridorModel(self.width, self.height, self.seed)
        
        self.weather = WeatherEngine(config)
        
        # Base static layers
        self.base_terrain_slope = None
        self.base_vegetation = None
        self.base_water = None
        self.human_density = None
        self.base_crop_density = None
        self.villages = []
        self.corridors = None
        
        # State tracking
        self.state_map = np.zeros((self.height, self.width), dtype=int)
        
    def generate_all(self):
        """Generates all stochastic layers and populates base matrices."""
        self.base_terrain_slope = self.terrain_gen.generate_slope_layer()
        self.base_vegetation = self.veg_gen.generate_vegetation()
        
        num_water = self.config['environment'].get('water_bodies', 10)
        num_rivers = self.config['environment'].get('river_count', 5)
        self.base_water = self.water_gen.generate_water(num_bodies=num_water, num_rivers=num_rivers)
        
        num_villages = self.config['environment'].get('village_count', 20)
        h_layer, c_layer, v_info = self.village_gen.generate_human_and_crop_layers(num_villages=num_villages)
        self.human_density = h_layer
        self.base_crop_density = c_layer
        self.villages = v_info
        
        num_corridors = self.config['environment'].get('corridor_count', 8)
        self.corridors = self.corridor_gen.generate_corridors(self.base_vegetation, num_corridors=num_corridors)
        
        # Split environment into 4 abstract macro "states" (e.g., State A, B, C, D) for migration tracking
        mid_y, mid_x = self.height // 2, self.width // 2
        self.state_map[:mid_y, :mid_x] = 0 # NW
        self.state_map[:mid_y, mid_x:] = 1 # NE
        self.state_map[mid_y:, :mid_x] = 2 # SW
        self.state_map[mid_y:, mid_x:] = 3 # SE
        
    def step_weather(self):
        """Advances weather state by 1 day."""
        self.weather.step_day()
        
    def get_features_at(self, x: int, y: int) -> dict:
        """Extract environment state at a given coordinate, applying dynamic weather modifiers."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ValueError(f"Coordinates ({x}, {y}) out of bounds")
            
        mods = self.weather.get_ecological_modifiers()
        
        # Dynamic reductions based on drought
        veg_val = self.base_vegetation[y, x] * mods['vegetation_multiplier']
        water_val = self.base_water[y, x] * mods['water_multiplier']
        crop_val = self.base_crop_density[y, x] * mods['vegetation_multiplier'] # Crops also die in drought
            
        return {
            'vegetation_density': veg_val,
            'crop_density': crop_val,
            'water_presence': water_val,
            'human_density': self.human_density[y, x],
            'terrain_slope': self.base_terrain_slope[y, x],
            'corridor_probability': self.corridors[y, x],
            'state_id': self.state_map[y, x],
            'heat_stress': mods['heat_stress_penalty']
        }
