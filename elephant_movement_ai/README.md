# RL Elephant Movement Ecosystem Simulation & Conflict Prediction

This repository hosts a research-grade simulation module designed to deploy Reinforcement Learning (RL) agents functioning as elephant herds across a dynamic, 200x200 simulated ecological grid. Utilizing PyTorch and Stable-Baselines3, the multi-agent system trains elephants to survive drought occurrences and resource scarcity while maximizing foraging rewards, organically leading to human-wildlife conflict events that are then captured to train machine learning risk predictors.

## Architectural Capabilities
1. **Dynamic Environment & Weather (`environment/`)**: Simulates 200x200 spatial arrays for dense vegetation, crop fields, static/river water systems, and villages. A `WeatherEngine` dynamically introduces droughts or storms over discrete days, directly scaling the availability of survival resources across the map.
2. **Gymnasium Multi-Agent RL Core (`agents/`)**: Elephant behaviors are modeled via Proximal Policy Optimization (PPO). The policy maps observation vectors (local surrounding features, health, memory grids, heat stress) to 9 discrete movement actions. 
3. **Complex Herd Dynamics (`agents/`)**: Captures non-linear organic movement events. Large herds fracture and split into new individual RL entities when scarcity/drought thresholds breach acceptable limits (`HerdSplitting`). Adult males sporadically defect to form rogue vectors with distinct, higher-tolerance risk matrices promoting crop damage (`SolitaryElephantTransition`).
4. **Interactions & Conflict Engine (`simulation/`)**: Negative rewards are actively broadcast by villages depending on time-of-day (active visual/audio defense vs passive windows). Collisions between RL agents and high `human_density` cells cast hard categorical events (`HOUSE_DAMAGE`, `CROP_RAIDING`) based on stochastic evaluation.
5. **Expanded Datasets (`dataset/`)**: Unifies the simulation iteration output into 5 unique CSV pipelines: `elephant_trajectories`, `conflict_events`, `crop_damage`, `migration_events`, and `weather_logs`.
6. **AI Predictor (`ml/`)**: Joins temporal weather trends against geo-telemetry to structure a rolling-window target feature for a `RandomForestClassifier`, tasked with calculating boolean predictive risk likelihoods of a village suffering conflict within 24 ahead simulation hours. 
7. **Visualization Toolkit (`visualization/`)**: Renders matplotlib layouts spanning base ecology markers, RL agent pathing arrays, cross-state regional drought-migration flow graphs, and ML-correlated Risk heatmaps.

## Configuration & Usage

`requirements.txt` includes spatial computation dependencies, `stable-baselines3`, `gymnasium`, and `torch`. 
Install globally or within a dedicated environment:

```bash
python -m pip install -r requirements.txt
```

To run the end-to-end sandbox—driving environment construction, active Model Training (PPO generation), Simulation Loop execution, Dataset publishing, ML fitting, and Visualization rendering:

```bash
python main.py
```

## Interactive "Snake-Game" Visualizer
To view a fast-forward, top-down animated run of the generated trajectories exactly like a game playing out over the map pixels, run:
```bash
python replay_simulation.py
```
*(Requires `pygame-ce` or `pygame` installed).*

All model checkpoints, extracted CSV tables, and mapping graphics are cleanly persisted to `output/`. Tuning parameters corresponding to Environment scale, Training epochs, and Reward Weights exist centrally in `config/simulation_config.yaml`.
