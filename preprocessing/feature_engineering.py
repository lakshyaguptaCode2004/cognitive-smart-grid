"""Feature Engineering for Smart Grid Data"""

import pandas as pd
import numpy as np

class FeatureEngineer:
    """Engineer features from raw smart grid data"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
    
    def add_cyclical_features(self):
        """Add cyclical time encodings"""
        self.df['hour_sin'] = np.sin(2 * np.pi * self.df['hour'] / 24)
        self.df['hour_cos'] = np.cos(2 * np.pi * self.df['hour'] / 24)
        self.df['dow_sin'] = np.sin(2 * np.pi * self.df['day_of_week'] / 7)
        self.df['dow_cos'] = np.cos(2 * np.pi * self.df['day_of_week'] / 7)
        return self
    
    def add_rolling_features(self, windows=[4, 24, 96]):
        """Add rolling statistics (4=1hr, 24=6hr, 96=24hr)"""
        for w in windows:
            self.df[f'load_roll_mean_{w}'] = self.df['load_kw'].rolling(w, min_periods=1).mean()
            self.df[f'load_roll_std_{w}'] = self.df['load_kw'].rolling(w, min_periods=1).std()
        return self
    
    def add_lag_features(self, lags=[1, 4, 96]):
        """Add lagged features"""
        for lag in lags:
            self.df[f'load_lag_{lag}'] = self.df['load_kw'].shift(lag)
        return self
    
    def handle_missing(self):
        """Fill missing values"""
        self.df = self.df.ffill().bfill()
        return self
    
    def engineer_all(self):
        """Run all feature engineering"""
        return (self
                .add_cyclical_features()
                .add_rolling_features()
                .add_lag_features()
                .handle_missing())
    
    def get_data(self):
        """Return engineered data"""
        return self.df
