class TimestepManager:
    """Manages simulation time progression and translates steps to time of day."""
    
    def __init__(self, steps_per_hour: int = 6):
        self.current_step = 0
        self.steps_per_hour = steps_per_hour
        
    def advance(self):
        """Advances the simulation by one timestep."""
        self.current_step += 1
        
    def get_hour_of_day(self) -> float:
        """Returns the hour of the day (0.0 to 24.0)."""
        total_hours = self.current_step / self.steps_per_hour
        return total_hours % 24
        
    def get_time_of_day_phase(self) -> str:
        """
        Translates current hour to a behavioral phase block.
        Day: 06:00-18:00
        Evening: 18:00-21:00
        Night: 21:00-03:00
        Morning: 03:00-06:00
        """
        hour = self.get_hour_of_day()
        
        if 6 <= hour < 18:
            return "day"
        elif 18 <= hour < 21:
            return "evening"
        elif 21 <= hour < 24 or 0 <= hour < 3:
            return "night"
        else: # 3 <= hour < 6
            return "morning"
