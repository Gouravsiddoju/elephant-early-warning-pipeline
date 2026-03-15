from typing import Tuple
from environment.environment import Environment
from agents.elephant_herd import ElephantHerd
from .movement_scoring import MovementScoring
from .probabilistic_selector import ProbabilisticSelector

class DecisionEngine:
    """Coordinates evaluating nearby cells and deciding the next herd position."""
    
    # 8 cardinal and ordinal directions + WAIT
    DIRECTIONS = {
        'N':  ( 0, -1),
        'NE': ( 1, -1),
        'E':  ( 1,  0),
        'SE': ( 1,  1),
        'S':  ( 0,  1),
        'SW': (-1,  1),
        'W':  (-1,  0),
        'NW': (-1, -1),
        'WAIT':(0,  0)
    }
    
    def __init__(self, config: dict):
        self.scoring = MovementScoring(config)
        self.selector = ProbabilisticSelector(seed=config['simulation'].get('random_seed', 42))
        
    def determine_next_move(self, env: Environment, herd: ElephantHerd, time_of_day: str) -> Tuple[int, int, str]:
        """
        Evaluates 8 neighboring cells + current cell.
        Returns (new_x, new_y, direction_string).
        """
        cx, cy = herd.leader_position
        candidate_moves = []
        
        for dir_name, (dx, dy) in self.DIRECTIONS.items():
            nx, ny = cx + dx, cy + dy
            
            # Check boundaries
            if 0 <= nx < env.width and 0 <= ny < env.height:
                features = env.get_features_at(nx, ny)
                mem_val = herd.memory_model.get_memory_value(nx, ny)
                
                score = self.scoring.compute_score(features, mem_val, time_of_day)
                
                candidate_moves.append({
                    'x': nx,
                    'y': ny,
                    'direction': dir_name,
                    'score': score
                })
                
        # Select stochastically
        chosen = self.selector.select_move(candidate_moves)
        return chosen['x'], chosen['y'], chosen['direction']
