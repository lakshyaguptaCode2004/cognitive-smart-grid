"""Complete Preprocessing Pipeline"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib
import os

try:
    from .feature_engineering import FeatureEngineer
except ImportError:
    from feature_engineering import FeatureEngineer

class PreprocessingPipeline:
    """Complete data preprocessing pipeline"""
    
    def __init__(self, data_path="./data/synthetic/smart_grid_data.csv"):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.feature_cols = None
    
    def load_data(self):
        """Load raw data"""
        df = pd.read_csv(self.data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"✓ Loaded: {df.shape[0]:,} samples, {df.shape[1]} features")
        return df
    
    def split_data(self, df, train_ratio=0.7, val_ratio=0.15):
        """Split into train/val/test"""
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        return {
            'train': df.iloc[:train_end],
            'val': df.iloc[train_end:val_end],
            'test': df.iloc[val_end:]
        }
    
    def run(self, output_dir="./data/processed"):
        """Run complete pipeline"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("PREPROCESSING PIPELINE")
        print("="*60 + "\n")
        
        # Load
        df = self.load_data()
        
        # Engineer features
        print("\nEngineering features...")
        engineer = FeatureEngineer(df)
        df_eng = engineer.engineer_all().get_data()
        print(f"✓ Engineered: {df_eng.shape[1]} total features")
        
        # Split
        print("\nSplitting data...")
        splits = self.split_data(df_eng)
        print(f"✓ Train: {len(splits['train']):,} samples")
        print(f"✓ Val: {len(splits['val']):,} samples")
        print(f"✓ Test: {len(splits['test']):,} samples")
        
        # Get feature columns
        self.feature_cols = [c for c in df_eng.columns 
                            if c not in ['timestamp', 'load_kw', 'pricing_tier']]
        
        # Normalize
        print("\nNormalizing features...")
        X_train = self.scaler.fit_transform(splits['train'][self.feature_cols])
        X_val = self.scaler.transform(splits['val'][self.feature_cols])
        X_test = self.scaler.transform(splits['test'][self.feature_cols])
        
        y_train = splits['train']['load_kw'].values
        y_val = splits['val']['load_kw'].values
        y_test = splits['test']['load_kw'].values
        
        # Save
        print("\nSaving processed data...")
        np.save(f"{output_dir}/X_train.npy", X_train)
        np.save(f"{output_dir}/X_val.npy", X_val)
        np.save(f"{output_dir}/X_test.npy", X_test)
        np.save(f"{output_dir}/y_train.npy", y_train)
        np.save(f"{output_dir}/y_val.npy", y_val)
        np.save(f"{output_dir}/y_test.npy", y_test)
        
        joblib.dump(self.scaler, f"{output_dir}/scaler.pkl")
        joblib.dump(self.feature_cols, f"{output_dir}/feature_cols.pkl")
        
        print(f"\n✓ Saved to: {output_dir}/")
        print(f"  Features: {len(self.feature_cols)}")
        print(f"  Train shape: {X_train.shape}")
        
        print("\n" + "="*60)
        print("PREPROCESSING COMPLETE")
        print("="*60)
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }

if __name__ == "__main__":
    pipeline = PreprocessingPipeline()
    pipeline.run()
