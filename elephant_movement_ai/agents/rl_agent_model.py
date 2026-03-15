import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

from environment.environment import Environment
from agents.rl_environment import ElephantRLEnv

class RLAgentModel:
    """Manages the training and inference of the Stable-Baselines3 PPO agent."""
    
    def __init__(self, config: dict, is_solitary: bool = False):
        self.config = config
        self.is_solitary = is_solitary
        self.model = None
        self.model_path = os.path.join(
            config['paths'].get('output_dir', 'output/'),
            f"ppo_elephant_{'solitary' if is_solitary else 'herd'}"
        )
        
    def train(self, global_env: Environment):
        """Trains the PPO agent on the provided environment."""
        print(f"Starting RL Training for {'Solitary' if self.is_solitary else 'Herd'} policy...")
        
        # Create a vectorized environment
        def make_env():
            return ElephantRLEnv(self.config, global_env, is_solitary=self.is_solitary)
            
        vec_env = make_vec_env(make_env, n_envs=4) # Train with 4 simultaneous environments
        
        rl_config = self.config.get('rl_training', {})
        total_timesteps = rl_config.get('total_timesteps', 10000)
        
        # Initialize PPO
        self.model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            learning_rate=rl_config.get('learning_rate', 0.0003),
            n_steps=rl_config.get('n_steps', 2048),
            batch_size=rl_config.get('batch_size', 64)
        )
        
        # Optional checkpointing
        checkpoint_dir = os.path.join(self.config['paths'].get('output_dir', 'output/'), 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_callback = CheckpointCallback(
            save_freq=max(1000, total_timesteps // 10),
            save_path=checkpoint_dir,
            name_prefix=f"rl_model_{'solitary' if self.is_solitary else 'herd'}"
        )
        
        self.model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
        self.model.save(self.model_path)
        print("Training complete and model saved.")
        
    def load(self):
        """Loads a pre-trained model."""
        if os.path.exists(self.model_path + ".zip"):
            self.model = PPO.load(self.model_path)
        else:
            raise FileNotFoundError(f"No trained model found at {self.model_path}")
            
    def predict(self, observation):
        """Inference step for the simulation loop."""
        if self.model is None:
            raise RuntimeError("Model is not loaded or trained yet.")
        action, _states = self.model.predict(observation, deterministic=True)
        return action
