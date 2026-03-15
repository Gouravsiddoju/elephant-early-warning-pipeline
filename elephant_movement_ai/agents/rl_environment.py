import gymnasium as gym
from gymnasium import spaces
import numpy as np

from environment.environment import Environment
from .reward_system import RewardSystem
from .memory_model import MemoryModel

# Actions enumeration
ACTION_WAIT = 0
ACTION_MOVE_N = 1
ACTION_MOVE_NE = 2
ACTION_MOVE_E = 3
ACTION_MOVE_SE = 4
ACTION_MOVE_S = 5
ACTION_MOVE_SW = 6
ACTION_MOVE_W = 7
ACTION_MOVE_NW = 8

class ElephantRLEnv(gym.Env):
    """
    OpenAI Gym compatible environment allowing RL algorithms to train Elephant policies.
    """
    
    def __init__(self, config: dict, env: Environment, min_x: int = 0, min_y: int = 0, is_solitary: bool = False):
        super(ElephantRLEnv, self).__init__()
        
        self.config = config
        self.global_env = env
        self.reward_system = RewardSystem(config)
        self.is_solitary = is_solitary
        
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = self.global_env.width - 1
        self.max_y = self.global_env.height - 1
        
        # Memory map bounds the visual memory
        self.memory = MemoryModel(self.global_env.width, self.global_env.height)
        
        # 9 discrete movement actions
        self.action_space = spaces.Discrete(9)
        
        # Observation space: 
        # [x, y, veg, water, crops, humans, slope, corridor, heat_stress, memory_val]
        # Bounded between appropriate limits (normalized roughly 0..1 for standard inputs)
        high = np.array([
            float(self.max_x), float(self.max_y), # positions
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.0, # features
            10.0 # max memory val
        ], dtype=np.float32)
        
        self.observation_space = spaces.Box(low=0.0, high=high, dtype=np.float32)
        
        self.current_pos = [self.min_x, self.min_y]
        self.step_count = 0
        self.max_steps = 2048 # Reset boundary for training episodes
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset position randomly in safe area
        valid_starts = np.argwhere(self.global_env.base_vegetation > 0.5)
        if len(valid_starts) > 0:
            idx = np.random.randint(0, len(valid_starts))
            self.current_pos = [valid_starts[idx][1], valid_starts[idx][0]]
        else:
            self.current_pos = [self.max_x // 2, self.max_y // 2]
            
        self.step_count = 0
        self.memory = MemoryModel(self.global_env.width, self.global_env.height)
        
        return self._get_obs(), {}
        
    def _get_obs(self):
        x, y = int(self.current_pos[0]), int(self.current_pos[1])
        feats = self.global_env.get_features_at(x, y)
        mem = self.memory.get_memory_value(x, y)
        
        # Normalize coordinates and memory to roughly 0..1 to prevent NN collapse
        norm_x = float(x) / float(self.max_x) if self.max_x > 0 else 0.0
        norm_y = float(y) / float(self.max_y) if self.max_y > 0 else 0.0
        norm_mem = min(1.0, mem / 10.0)
        
        obs = np.array([
            norm_x, norm_y,
            feats['vegetation_density'],
            feats['water_presence'],
            feats['crop_density'],
            feats['human_density'],
            feats['terrain_slope'],
            feats['corridor_probability'],
            feats['heat_stress'] / 5.0, # Approximate normalize
            norm_mem
        ], dtype=np.float32)
        return obs

    def step(self, action: int):
        self.step_count += 1
        x, y = int(self.current_pos[0]), int(self.current_pos[1])
        
        # Apply empirical resting probability
        rest_prob = self.config.get('elephant', {}).get('rest_probability', 0.0)
        if np.random.random() < rest_prob:
            action = ACTION_WAIT
            
        # Translate action to movement
        dx, dy = 0, 0
        if action == ACTION_MOVE_N:  dy = -1
        elif action == ACTION_MOVE_NE: dx, dy = 1, -1
        elif action == ACTION_MOVE_E:  dx = 1
        elif action == ACTION_MOVE_SE: dx, dy = 1, 1
        elif action == ACTION_MOVE_S:  dy = 1
        elif action == ACTION_MOVE_SW: dx, dy = -1, 1
        elif action == ACTION_MOVE_W:  dx = -1
        elif action == ACTION_MOVE_NW: dx, dy = -1, -1
        
        # Override network actions if hugging a wall to force them inwards
        margin = 10
        if x <= margin and dx < 0:
            dx = 1 # Force East
        elif x >= self.max_x - margin and dx > 0:
            dx = -1 # Force West
            
        if y <= margin and dy < 0:
            dy = 1 # Force South
        elif y >= self.max_y - margin and dy > 0:
            dy = -1 # Force North
            
        nx = np.clip(x + dx, self.min_x, self.max_x)
        ny = np.clip(y + dy, self.min_y, self.max_y)
        self.current_pos = [nx, ny]
        
        # Get features and calculate reward
        feats = self.global_env.get_features_at(int(nx), int(ny))
        reward = self.reward_system.calculate_step_reward(feats, action, self.is_solitary, nx, ny, self.global_env.width, self.global_env.height)
        
        # Update memory
        mem_update = (feats['crop_density'] * 2.0) + (feats['water_presence'] * 5.0)
        self.memory.update_memory(int(nx), int(ny), mem_update)
        
        terminated = False
        truncated = False
        if self.step_count >= self.max_steps:
            truncated = True
            
        # Give a small negative reward for doing nothing (encourage exploration unless very hot)
        if action == ACTION_WAIT and feats['heat_stress'] < 0.2:
            reward -= 0.5
            
        info = {'features': feats}
        return self._get_obs(), reward, terminated, truncated, info
