import os
from .trajectory_dataset import TrajectoryDataset
from .conflict_events_dataset import ConflictEventsDataset
from .crop_damage_dataset import CropDamageDataset
from .migration_dataset import MigrationDataset
from .weather_dataset import WeatherDataset

class DatasetBuilder:
    """Orchestrates serialization of all 5 simulation output datasets."""
    
    def __init__(self, config: dict):
        self.output_dir = config['paths'].get('datasets_dir', 'output/datasets/')
        
        self.trajectory_ds = TrajectoryDataset(self.output_dir)
        self.conflict_ds = ConflictEventsDataset(self.output_dir)
        self.crop_ds = CropDamageDataset(self.output_dir)
        self.migration_ds = MigrationDataset(self.output_dir)
        self.weather_ds = WeatherDataset(self.output_dir)
        
    def build_datasets(self, sim_engine):
        """Pass the simulation engine and tell it to dump all logged arrays to disk."""
        print(f"\nBuilding and saving datasets to {self.output_dir}...")
        self.trajectory_ds.save(sim_engine.logs_trajectories)
        self.conflict_ds.save(sim_engine.logs_conflicts)
        self.crop_ds.save(sim_engine.logs_crops)
        self.migration_ds.save(sim_engine.logs_migration)
        self.weather_ds.save(sim_engine.logs_weather)
        print(f"All datasets saved successfully.")
