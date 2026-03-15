import uuid
import numpy as np

class SolitaryElephantTransition:
    """Handles an adult male occasionally leaving a herd to become solitary."""
    
    def __init__(self, config: dict):
        self.rng = np.random.default_rng(config['simulation'].get('random_seed', 42))
        # Hardcode a low probability per timestep (approx 0.05% per 10 min)
        self.solitary_chance = 0.0005
        
    def check_for_solitary_male(self, herd) -> list:
        """
        Rolls a long-shot chance for an adult male to leave a sizable herd organically.
        """
        new_solitary_agents = []
        if herd.herd_size > 5 and self.rng.random() < self.solitary_chance:
            herd.herd_size -= 1
            new_solitary_agents.append({
                'id': str(uuid.uuid4())[:8],
                'start_x': herd.leader_position[0],
                'start_y': herd.leader_position[1],
                'size': 1,
                'is_solitary': True
            })
            print(f"An Adult Male departed Herd {herd.herd_id} to become a Solitary Agent.")
            
        return new_solitary_agents
