"""
Demand Response RL Environment
Custom Gymnasium environment for demand response optimization
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

class DemandResponseEnv(gym.Env):
    """
    Demand Response Environment
    
    State: [load, forecast, price, carbon, hour, weekend, comfort_score]
    Action: [reduce_hvac%, delay_ev, shift_appliance, activate_battery, do_nothing]
    Reward: -peak_penalty - cost_penalty - carbon_penalty - comfort_violation + savings
    """
    
    def __init__(self, load_data, price_data, carbon_data, max_steps=96):
        super().__init__()
        
        self.load_data = load_data
        self.price_data = price_data
        self.carbon_data = carbon_data
        self.max_steps = max_steps
        
        # State space: 7 dimensions
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0]),
            high=np.array([2000, 2000, 1, 1000, 23, 1, 1]),
            dtype=np.float32
        )
        
        # Action space: 5 discrete actions
        self.action_space = spaces.Discrete(5)
        
        self.current_step = 0
        self.current_load = 0
        self.baseline_load = 0
        self.peak_threshold = np.percentile(load_data, 95)
        
    def reset(self, seed=None, options=None):
        """Reset environment"""
        super().reset(seed=seed)
        
        self.current_step = np.random.randint(0, len(self.load_data) - self.max_steps)
        self.current_load = self.load_data[self.current_step]
        self.baseline_load = self.current_load
        
        state = self._get_state()
        info = {}
        
        return state, info
    
    def _get_state(self):
        """Get current state"""
        idx = self.current_step
        
        # Forecast (next timestep load)
        forecast = self.load_data[min(idx + 1, len(self.load_data) - 1)]
        
        # Price (normalized to 0-1)
        price = self.price_data[idx] / 0.25
        
        # Carbon (normalized)
        carbon = self.carbon_data[idx] / 800
        
        # Hour of day
        hour = idx % 96 // 4
        
        # Weekend indicator
        weekend = 1 if (idx // 96) % 7 >= 5 else 0
        
        # Comfort score (random for now, would come from NLP in full system)
        comfort = np.random.uniform(0.3, 0.9)
        
        return np.array([
            self.current_load,
            forecast,
            price,
            carbon,
            hour,
            weekend,
            comfort
        ], dtype=np.float32)
    
    def step(self, action):
        """Take action and return next state"""
        
        # Action effects
        load_reduction = 0
        comfort_violation = 0
        
        if action == 0:  # Reduce HVAC 20%
            load_reduction = self.current_load * 0.20
            comfort_violation = 0.1
        elif action == 1:  # Delay EV
            load_reduction = self.current_load * 0.15
            comfort_violation = 0.05
        elif action == 2:  # Shift appliance
            load_reduction = self.current_load * 0.10
            comfort_violation = 0.02
        elif action == 3:  # Activate battery
            load_reduction = self.current_load * 0.25
            comfort_violation = 0.0
        else:  # Do nothing
            load_reduction = 0
            comfort_violation = 0
        
        # Update load
        self.current_load = max(self.current_load - load_reduction, 50)
        
        # Calculate rewards
        peak_penalty = max(0, (self.current_load - self.peak_threshold) / 100) * -5
        cost_penalty = (self.current_load * self.price_data[self.current_step]) * -0.01
        carbon_penalty = (self.current_load * self.carbon_data[self.current_step] / 1000) * -0.1
        comfort_penalty = comfort_violation * -10
        savings_bonus = load_reduction * 0.05
        
        reward = peak_penalty + cost_penalty + carbon_penalty + comfort_penalty + savings_bonus
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= min(len(self.load_data) - 1, self.max_steps)
        truncated = False
        
        state = self._get_state()
        info = {
            'load': self.current_load,
            'reduction': load_reduction,
            'peak_penalty': peak_penalty,
            'cost_penalty': cost_penalty,
            'carbon_penalty': carbon_penalty
        }
        
        return state, reward, done, truncated, info
    
    def render(self):
        """Render environment"""
        print(f"Step: {self.current_step}, Load: {self.current_load:.1f} kW")

if __name__ == "__main__":
    # Load data
    import pandas as pd
    
    df = pd.read_csv('./data/synthetic/smart_grid_data.csv')
    
    env = DemandResponseEnv(
        load_data=df['load_kw'].values,
        price_data=df['price_per_kwh'].values,
        carbon_data=df['carbon_intensity'].values
    )
    
    # Test environment
    print("Testing Demand Response Environment...")
    state, info = env.reset()
    print(f"Initial state: {state}")
    
    for _ in range(10):
        action = env.action_space.sample()
        state, reward, done, truncated, info = env.step(action)
        print(f"Action: {action}, Reward: {reward:.2f}, Load: {info['load']:.1f} kW")
        
        if done:
            break
    
    print("✓ Environment test complete!")
