import numpy as np

class ConflictEventGenerator:
    """Generates discrete conflict records when an RL agent intersects human resources."""
    
    def __init__(self, config: dict):
        self.rng = np.random.default_rng(config['simulation'].get('random_seed', 42))
        
    def evaluate_herd_position(self, env, herd_x: int, herd_y: int, herd_size: int, is_solitary: bool) -> list:
        """
        Checks if the current location explicitly sparks a registrable event.
        Returns a list of event dictionaries.
        """
        events = []
        
        # 1. House/Human Damage checks (High human density zone)
        h_dens = env.human_density[int(herd_y), int(herd_x)]
        if h_dens > 0.4:
            # High chance of negative encounter
            prob = 0.2 if is_solitary else 0.05
            if self.rng.random() < prob:
                events.append({
                    'event_type': 'HOUSE_DAMAGE' if self.rng.random() > 0.3 else 'HUMAN_ENCOUNTER',
                    'damage_score': self.rng.uniform(0.5, 1.0) * herd_size,
                    'is_solitary': is_solitary
                })
                
        # 2. Crop Raiding (Crop density > 0 and close to village)
        c_dens = env.base_crop_density[int(herd_y), int(herd_x)]
        if c_dens > 0.2:
            prob = 0.8 if is_solitary else 0.4
            if self.rng.random() < prob:
                 events.append({
                    'event_type': 'CROP_RAIDING',
                    'damage_score': c_dens * herd_size * self.rng.uniform(0.8, 1.2),
                    'is_solitary': is_solitary
                })
                
        return events
