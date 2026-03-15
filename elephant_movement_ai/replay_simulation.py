import pandas as pd
import numpy as np
import yaml
import sys
import os

try:
    import pygame
except ImportError:
    print("Pygame is not installed. Please run: pip install pygame")
    sys.exit(1)

from environment.environment import Environment

os.environ['SDL_VIDEO_CENTERED'] = '1'

def load_config(config_path="config/simulation_config.yaml") -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def run_replay():
    print("Loading simulation data for visual replay...")
    config = load_config()
    
    # Load environment map
    env = Environment(config)
    env.generate_all()
    
    # Load trajectories
    traj_path = os.path.join(config['paths'].get('datasets_dir', 'output/datasets/'), "elephant_trajectories.csv")
    if not os.path.exists(traj_path):
        print("No trajectory data found. Please run main.py first to generate data.")
        return

    df = pd.read_csv(traj_path)
    if df.empty:
        print("Trajectory dataset is empty.")
        return

    pygame.init()
    
    cell_size = 4
    width = env.width * cell_size
    height = env.height * cell_size
    
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Elephant Simulation Playback")
    
    # Pre-render background
    print("Rendering background map...")
    bg_surface = pygame.Surface((width, height))
    bg_surface.fill((34, 139, 34)) # Base Forest Green
    
    # Draw water only
    for y in range(env.height):
        for x in range(env.width):
            water = env.base_water[y, x]
            if water > 0.1:
                pygame.draw.rect(bg_surface, (30, 144, 255), (x * cell_size, height - (y + 1) * cell_size, cell_size, cell_size)) # Dodger Blue
    # Draw villages
    for v in env.villages:
        vx_px = int(v['x'] * cell_size)
        vy_px = int(height - v['y'] * cell_size)
        # Draw a simple brown hut marker
        pygame.draw.rect(bg_surface, (139, 69, 19), (vx_px - 4, vy_px - 4, 8, 8))

    # Pygame loop setup
    clock = pygame.time.Clock()
    timesteps = sorted(df['timestep'].unique())
    fps = 30
    
    running = True
    step_idx = 0
    font = pygame.font.SysFont("Arial", 24, bold=True)
    
    print("Starting Playback! Close the window to exit.")
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        if step_idx < len(timesteps):
            ts = timesteps[step_idx]
            current_actors = df[df['timestep'] == ts]
            
            screen.blit(bg_surface, (0, 0))
            
            # Draw agents
            for _, actor in current_actors.iterrows():
                # Correct Y axis orientation for pygame (and scale by cell size)
                ax_px = int(actor['x']) * cell_size
                # Use standard logic: top-left is 0,0, map y=0 is bottom
                ay_px = height - (int(actor['y']) * cell_size) - cell_size
                
                is_solitary = False
                try:
                    if 'herd_id' in actor:
                        herd_str = str(actor['herd_id'])
                        if "NONE" in herd_str or herd_str == 'nan':
                            is_solitary = True
                except Exception:
                    pass
                
                if is_solitary:
                    color = (255, 140, 0) # Orange for lone rogue elephants
                    radius = max(2, cell_size // 2)
                else:
                    color = (255, 215, 0) # Yellow for herds
                    radius = max(3, cell_size)
                    
                center_x = ax_px + cell_size//2
                center_y = ay_px + cell_size//2
                pygame.draw.circle(screen, color, (center_x, center_y), radius)
                # Outer black ring for visibility
                pygame.draw.circle(screen, (0, 0, 0), (center_x, center_y), radius, 1)

            day = ts // 144
            time_txt = font.render(f"Day: {day} | Timestep: {ts}", True, (255, 255, 255))
            agents_txt = font.render(f"Active Agents: {len(current_actors)}", True, (255, 200, 0))
            
            # Draw semi-transparent background for text
            txt_bg = pygame.Surface((250, 70))
            txt_bg.set_alpha(180)
            txt_bg.fill((0, 0, 0))
            screen.blit(txt_bg, (10, 10))
            
            screen.blit(time_txt, (20, 20))
            screen.blit(agents_txt, (20, 50))
            
            pygame.display.flip()
            # Fast forward slightly so 10,000 steps doesn't take hours
            step_idx += 5
            clock.tick(fps)
        else:
            # Replay ended (Keep showing last frame instead of exiting immediately)
            end_font = pygame.font.SysFont("Arial", 48, bold=True)
            end_txt = end_font.render("Simulation Replay Complete", True, (255, 255, 255))
            screen.blit(end_txt, (width//2 - end_txt.get_width()//2, height//2 - end_txt.get_height()//2))
            pygame.display.flip()
            
            # Just wait for quit event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            clock.tick(10)
            
    pygame.quit()

if __name__ == "__main__":
    run_replay()
