import numpy as np

class HumanInteraction:
    """Calculates active disturbance around villages that heavily drops RL rewards."""
    
    def __init__(self, config: dict):
        self.rng = np.random.default_rng(config['simulation'].get('random_seed', 42))
        
    def calculate_disturbance(self, env, herd_x: int, herd_y: int, time_phase: str) -> float:
        """
        Calculates a dynamic scalar penalty.
        Villages are active during the day/evening (noise, firecrackers, vehicles).
        Quiet at night, letting elephants slip into crops easier.
        """
        base_human_density = env.human_density[int(herd_y), int(herd_x)]
        if base_human_density <= 0.05:
            return 0.0 # No meaningful disturbance
            
        disturbance = base_human_density * 5.0 # Scaler
        
        # Temporal modulation
        if time_phase in ["day", "evening"]:
            # Active defense (high disturbance)
            disturbance *= self.rng.uniform(1.5, 2.5) 
        else:
            # Passive defense (night/morning)
            disturbance *= self.rng.uniform(0.5, 1.0)
            
        return disturbance
