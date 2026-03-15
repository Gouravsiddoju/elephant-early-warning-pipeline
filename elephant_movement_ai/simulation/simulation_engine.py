import numpy as np
import time
from typing import List, Dict

from environment.environment import Environment
from agents.rl_agent_model import RLAgentModel
from agents.herd_dynamics import HerdDynamics
from agents.herd_splitting import HerdSplitting
from agents.solitary_elephant import SolitaryElephantTransition

from simulation.timestep_manager import TimestepManager
from simulation.human_interaction import HumanInteraction
from simulation.conflict_event_generator import ConflictEventGenerator

# We simulate "Elephants" which are essentially instances storing state, while the PPO models are global
class ElephantInstance:
    def __init__(self, h_id, x, y, size, is_solitary=False):
        self.herd_id = h_id
        self.leader_position = [x, y]
        self.herd_size = size
        self.is_solitary = is_solitary
        from agents.memory_model import MemoryModel
        # Need passing dim width/height
        self.memory = None 

class SimulationEngine:
    """Core loop executing the MARL model over discrete timesteps."""
    
    def __init__(self, config: dict):
        self.config = config
        self.rng = np.random.default_rng(config['simulation'].get('random_seed', 42))
        
        self.env = Environment(config)
        self.timestep_manager = TimestepManager(
            steps_per_hour=config['simulation'].get('steps_per_hour', 6)
        )
        
        # RL Models (Policies)
        self.herd_model = RLAgentModel(config, is_solitary=False)
        self.solitary_model = RLAgentModel(config, is_solitary=True)
        
        # Dynamics
        self.herd_dynamics = HerdDynamics(config)
        self.herd_splitting = HerdSplitting(config)
        self.solitary_transition = SolitaryElephantTransition(config)
        
        # Interactions
        self.human_interaction = HumanInteraction(config)
        self.conflict_gen = ConflictEventGenerator(config)
        
        self.active_agents: List[ElephantInstance] = []
        
        # Loggers
        self.logs_trajectories = []
        self.logs_conflicts = []
        self.logs_crops = []
        self.logs_migration = []
        self.logs_weather = []
        
    def setup(self, pretrain: bool = True):
        """Prepare environment, instantiate models, and spawn."""
        print("Generating 200x200 environment layers...")
        self.env.generate_all()
        
        if pretrain:
            # Train briefly or load
            try:
                self.herd_model.load()
                self.solitary_model.load()
                print("Loaded pre-trained RL models.")
            except FileNotFoundError:
                print("No pre-trained models found. Running mini-training batches...")
                self.herd_model.train(self.env)
                self.solitary_model.train(self.env)
        
        # Spawn initial herds
        num_herds = self.config['herd'].get('count', 3)
        initial_size = self.config['herd'].get('initial_size', 20)
        
        from agents.memory_model import MemoryModel
        for h_id in range(num_herds):
            forest_points = np.argwhere(self.env.base_vegetation > 0.5)
            if len(forest_points) > 0:
                idx = self.rng.integers(0, len(forest_points))
                sy, sx = forest_points[idx]
            else:
                sx, sy = 100, 100
                
            agent = ElephantInstance(h_id=f"H_{h_id}", x=int(sx), y=int(sy), size=initial_size)
            agent.memory = MemoryModel(self.env.width, self.env.height)
            self.active_agents.append(agent)

    def _build_obs_vector(self, agent: ElephantInstance) -> np.ndarray:
        x, y = int(agent.leader_position[0]), int(agent.leader_position[1])
        feats = self.env.get_features_at(x, y)
        mem = agent.memory.get_memory_value(x, y)
        # Match gym shape
        return np.array([
            float(x), float(y),
            feats['vegetation_density'],
            feats['water_presence'],
            feats['crop_density'],
            feats['human_density'],
            feats['terrain_slope'],
            feats['corridor_probability'],
            feats['heat_stress'],
            mem
        ], dtype=np.float32)

    def run(self):
        duration_days = self.config['simulation'].get('duration_days', 90) # longer for RL
        steps_per_hour = self.config['simulation'].get('steps_per_hour', 6)
        total_steps = duration_days * 24 * steps_per_hour
        
        print(f"Starting MARL simulation for {total_steps} timesteps ({duration_days} days)...")
        start_time = time.time()
        
        for step in range(total_steps):
            time_phase = self.timestep_manager.get_time_of_day_phase()
            
            # 1. Weather ticks once per day
            if step > 0 and step % (steps_per_hour * 24) == 0:
                self.env.step_weather()
                # Log weather
                self.logs_weather.append({
                    'timestep': step,
                    'temperature': self.env.weather.current_temperature,
                    'rainfall': self.env.weather.current_rainfall,
                    'drought_index': self.env.weather.drought_index
                })
                
            new_agents_queue = []
                
            # 2. Iterate Agents
            for agent in self.active_agents:
                x, y = int(agent.leader_position[0]), int(agent.leader_position[1])
                old_state = self.env.state_map[y, x]
                
                # Fetch Observation
                obs = self._build_obs_vector(agent)
                
                # Predict Action using generic policy
                model = self.solitary_model if agent.is_solitary else self.herd_model
                action = model.predict(obs) # Returns scalar integer
                
                # Apply movement logic (0-8)
                dx, dy = 0, 0
                if action == 1:  dy = -1
                elif action == 2: dx, dy = 1, -1
                elif action == 3:  dx = 1
                elif action == 4: dx, dy = 1, 1
                elif action == 5:  dy = 1
                elif action == 6: dx, dy = -1, 1
                elif action == 8: dx, dy = -1, -1
                
                # Override network actions if hugging a wall to force them inwards
                margin = 10
                if x <= margin and dx < 0:
                    dx = 1 # Force East
                elif x >= self.env.width - 1 - margin and dx > 0:
                    dx = -1 # Force West
                    
                if y <= margin and dy < 0:
                    dy = 1 # Force South
                elif y >= self.env.height - 1 - margin and dy > 0:
                    dy = -1 # Force North
                
                nx = np.clip(x + dx, 0, self.env.width - 1)
                ny = np.clip(y + dy, 0, self.env.height - 1)
                agent.leader_position = [nx, ny]
                
                # Extract state features at new pos
                feats = self.env.get_features_at(nx, ny)
                
                # Update agent memory explicitly
                mem_update = (feats['crop_density'] * 2.0) + (feats['water_presence'] * 5.0)
                agent.memory.update_memory(nx, ny, mem_update)
                
                new_state = self.env.state_map[ny, nx]
                
                # Record Trajectory
                self.logs_trajectories.append({
                    'timestep': step,
                    'elephant_id': agent.herd_id, # proxy for herd
                    'herd_id': agent.herd_id if not agent.is_solitary else 'NONE',
                    'x': nx,
                    'y': ny,
                    'state': new_state
                })
                
                # Migration check
                if old_state != new_state:
                    self.logs_migration.append({
                        'timestep': step,
                        'herd_id': agent.herd_id,
                        'from_state': old_state,
                        'to_state': new_state
                    })
                    
                # Conflict tracking
                events = self.conflict_gen.evaluate_herd_position(self.env, nx, ny, agent.herd_size, agent.is_solitary)
                for ev in events:
                    # Find closest village ID
                    v_dists = [(v['id'], np.hypot(nx-v['x'], ny-v['y'])) for v in self.env.villages]
                    v_id = min(v_dists, key=lambda i: i[1])[0] if v_dists else -1
                    
                    if ev['event_type'] == 'CROP_RAIDING':
                        self.logs_crops.append({
                            'timestep': step,
                            'village': v_id,
                            'crop_damage': ev['damage_score'],
                            'herd_size': agent.herd_size
                        })
                    else:
                        self.logs_conflicts.append({
                            'timestep': step,
                            'village_id': v_id,
                            'event_type': ev['event_type'],
                            'herd_size': agent.herd_size,
                            'damage_score': ev['damage_score']
                        })
                
                # Herd Dynamics Processing
                if not agent.is_solitary:
                    # 1. Check Solitary departing
                    departing_males = self.solitary_transition.check_for_solitary_male(agent)
                    for dm in departing_males:
                        m_agent = ElephantInstance(h_id=f"M_{dm['id']}", x=dm['start_x'], y=dm['start_y'], size=1, is_solitary=True)
                        from agents.memory_model import MemoryModel
                        m_agent.memory = MemoryModel(self.env.width, self.env.height)
                        # Inherit portions of herd memory
                        m_agent.memory.memory_map += (agent.memory.memory_map * 0.5)
                        new_agents_queue.append(m_agent)
                        
                    # 2. Check Herd Fracturing
                    splinters = self.herd_splitting.check_and_split(agent, self.env.weather.drought_index, feats)
                    for sh in splinters:
                        s_agent = ElephantInstance(h_id=f"H_SPLIT_{sh['id']}", x=sh['start_x'], y=sh['start_y'], size=sh['size'], is_solitary=False)
                        from agents.memory_model import MemoryModel
                        s_agent.memory = MemoryModel(self.env.width, self.env.height)
                        s_agent.memory.memory_map += agent.memory.memory_map
                        new_agents_queue.append(s_agent)
            
            self.active_agents.extend(new_agents_queue)
            self.timestep_manager.advance()
            
            if step > 0 and step % (steps_per_hour * 24 * 5) == 0:
                print(f"[{step}/{total_steps}] Completed Day {step // (steps_per_hour * 24)} | Active Agents: {len(self.active_agents)}")
                
        print(f"Simulation complete in {(time.time() - start_time):.2f}s.")
