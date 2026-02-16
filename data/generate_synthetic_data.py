"""
Synthetic Smart Grid Dataset Generator
Generates realistic 365-day electricity, weather, pricing, carbon, and user feedback data
"""

import numpy as np
import pandas as pd
import random
import os
from typing import Tuple, List

class SmartGridDataGenerator:
    """Generate realistic synthetic smart grid data"""
    
    def __init__(self, start_date="2023-01-01", num_days=365, freq_min=15, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        
        self.start_date = pd.to_datetime(start_date)
        self.num_days = num_days
        self.freq_min = freq_min
        self.timestamps = pd.date_range(
            start=self.start_date,
            periods=(num_days * 24 * 60) // freq_min,
            freq=f'{freq_min}min'
        )
        print(f"✓ Initialized: {len(self.timestamps):,} timestamps")
    
    def generate_load(self) -> np.ndarray:
        """Generate electricity load with realistic patterns"""
        n = len(self.timestamps)
        load = np.zeros(n)
        
        for i, ts in enumerate(self.timestamps):
            base = 500  # Base 500 kW
            hour = ts.hour + ts.minute / 60
            
            # Daily pattern
            if 6 <= hour < 9:
                daily = 300 * np.sin((hour - 6) * np.pi / 3)
            elif 17 <= hour < 22:
                daily = 450 * np.sin((hour - 17) * np.pi / 5)
            elif hour >= 22 or hour < 6:
                daily = -100
            else:
                daily = 100 * np.sin(hour * np.pi / 12)
            
            weekly = -100 if ts.dayofweek >= 5 else 0
            seasonal = 200 * abs(np.cos(2 * np.pi * ts.dayofyear / 365))
            noise = np.random.normal(0, 30)
            spike = np.random.uniform(200, 500) if np.random.random() < 0.02 else 0
            
            load[i] = max(base + daily + weekly + seasonal + noise + spike, 100)
        
        print(f"✓ Load: μ={load.mean():.1f} kW, σ={load.std():.1f} kW")
        return load
    
    def generate_temperature(self) -> np.ndarray:
        """Generate temperature"""
        temp = np.zeros(len(self.timestamps))
        
        for i, ts in enumerate(self.timestamps):
            avg_temp = 15 + 15 * np.sin(2 * np.pi * (ts.dayofyear - 80) / 365)
            hour = ts.hour + ts.minute / 60
            daily_var = 5 * np.sin((hour - 6) * np.pi / 12)
            temp[i] = avg_temp + daily_var + np.random.normal(0, 2)
        
        print(f"✓ Temperature: μ={temp.mean():.1f}°C")
        return temp
    
    def generate_humidity(self, temp: np.ndarray) -> np.ndarray:
        """Generate humidity"""
        humidity = np.clip(70 - 1.5 * (temp - 15) + np.random.normal(0, 5, len(temp)), 20, 95)
        print(f"✓ Humidity: μ={humidity.mean():.1f}%")
        return humidity
    
    def generate_pricing(self) -> Tuple[np.ndarray, List[str]]:
        """Generate TOU pricing"""
        prices = np.zeros(len(self.timestamps))
        tiers = []
        
        for i, ts in enumerate(self.timestamps):
            hour = ts.hour
            if 23 <= hour or hour < 7:
                price, tier = 0.08, 'off-peak'
            elif 17 <= hour < 22:
                price, tier = 0.20, 'on-peak'
            else:
                price, tier = 0.12, 'mid-peak'
            
            prices[i] = price + np.random.normal(0, 0.005)
            tiers.append(tier)
        
        print(f"✓ Pricing: off=$0.08, mid=$0.12, on=$0.20")
        return prices, tiers
    
    def generate_carbon(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate carbon intensity"""
        carbon = np.zeros(len(self.timestamps))
        renewable = np.zeros(len(self.timestamps))
        
        for i, ts in enumerate(self.timestamps):
            hour = ts.hour
            if 7 <= hour <= 19:
                solar = np.sin((hour - 7) * np.pi / 12)
                renew_pct = 20 + 40 * solar
            else:
                renew_pct = 15
            
            carbon[i] = max(600 * (1 - renew_pct / 100) + np.random.normal(0, 20), 150)
            renewable[i] = np.clip(renew_pct + np.random.normal(0, 3), 5, 80)
        
        print(f"✓ Carbon: μ={carbon.mean():.0f} gCO2/kWh")
        return carbon, renewable
    
    def generate_user_feedback(self, n=10000) -> pd.DataFrame:
        """Generate user feedback"""
        templates = {
            'comfort': ["Don't adjust {app}", "Keep {app} on"],
            'cost': ["Minimize bill", "Shift {app} to off-peak"],
            'env': ["Use renewable energy", "Reduce carbon"],
            'flex': ["{app} can be delayed", "Flexible with {app}"]
        }
        
        apps = ['EV', 'HVAC', 'washer', 'dryer']
        data = []
        
        for _ in range(n):
            cat = random.choice(list(templates.keys()))
            text = random.choice(templates[cat]).format(app=random.choice(apps))
            
            data.append({
                'text': text,
                'category': cat,
                'sentiment': np.clip({'comfort': 0.3, 'cost': 0.6, 'env': 0.7, 'flex': 0.8}[cat] + np.random.normal(0, 0.1), 0, 1),
                'flexibility': np.clip({'comfort': 0.2, 'cost': 0.7, 'env': 0.6, 'flex': 0.9}[cat] + np.random.normal(0, 0.1), 0, 1)
            })
        
        print(f"✓ User feedback: {n:,} samples")
        return pd.DataFrame(data)
    
    def save_all(self, output_dir="./data/synthetic"):
        """Generate and save datasets"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("GENERATING DATASETS")
        print("="*60 + "\n")
        
        load = self.generate_load()
        temp = self.generate_temperature()
        humidity = self.generate_humidity(temp)
        prices, tiers = self.generate_pricing()
        carbon, renewable = self.generate_carbon()
        
        df = pd.DataFrame({
            'timestamp': self.timestamps,
            'load_kw': load,
            'temperature_c': temp,
            'humidity_pct': humidity,
            'price_per_kwh': prices,
            'pricing_tier': tiers,
            'carbon_intensity': carbon,
            'renewable_pct': renewable,
            'hour': self.timestamps.hour,
            'day_of_week': self.timestamps.dayofweek,
            'is_weekend': (self.timestamps.dayofweek >= 5).astype(int),
        })
        
        df.to_csv(f"{output_dir}/smart_grid_data.csv", index=False)
        print(f"\n✓ Saved: smart_grid_data.csv ({df.shape[0]:,} rows)")
        
        feedback = self.generate_user_feedback()
        feedback.to_csv(f"{output_dir}/user_feedback.csv", index=False)
        print(f"✓ Saved: user_feedback.csv ({feedback.shape[0]:,} rows)")
        
        print("\n" + "="*60)
        print("COMPLETE")
        print("="*60)

if __name__ == "__main__":
    gen = SmartGridDataGenerator()
    gen.save_all()
