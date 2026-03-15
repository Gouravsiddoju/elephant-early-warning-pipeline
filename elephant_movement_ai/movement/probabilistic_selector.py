import numpy as np
from typing import List, Tuple

class ProbabilisticSelector:
    """Uses softmax probability to select an action logically rather than deterministically."""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        
    def select_move(self, moves: List[dict]) -> dict:
        """
        Selects a move from a list of dictionaries where each dict has at least:
        - 'score': float
        Returns the selected move dict.
        """
        if not moves:
            raise ValueError("No moves provided")
            
        scores = np.array([m['score'] for m in moves])
        
        # Softmax computation
        # Subtract max for numerical stability
        exp_scores = np.exp(scores - np.max(scores))
        probabilities = exp_scores / np.sum(exp_scores)
        
        # Select index based on probabilities
        chosen_index = self.rng.choice(len(moves), p=probabilities)
        return moves[chosen_index]
