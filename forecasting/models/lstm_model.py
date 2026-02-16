"""LSTM Model for Load Forecasting"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import joblib

class LSTMForecaster(nn.Module):
    """LSTM with Attention for load forecasting"""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        """Forward pass"""
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        pred = self.fc(last_hidden)
        return pred

class LSTMTrainer:
    """Trainer for LSTM"""
    
    def __init__(self, model, lr=0.001, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.train_losses = []
        self.val_losses = []
    
    def create_sequences(self, X, y, seq_length=24):
        """Create sequences for time series"""
        X_seq, y_seq = [], []
        
        for i in range(len(X) - seq_length):
            X_seq.append(X[i:i+seq_length])
            y_seq.append(y[i+seq_length])
        
        return np.array(X_seq), np.array(y_seq)
    
    def train_epoch(self, train_loader):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            self.optimizer.zero_grad()
            pred = self.model(X_batch)
            loss = self.criterion(pred.squeeze(), y_batch)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                pred = self.model(X_batch)
                loss = self.criterion(pred.squeeze(), y_batch)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=64, seq_length=24):
        """Train model"""
        print("\n" + "="*60)
        print("TRAINING LSTM FORECASTER")
        print("="*60)
        
        # Create sequences
        print(f"\nCreating sequences (length={seq_length})...")
        X_train_seq, y_train_seq = self.create_sequences(X_train, y_train, seq_length)
        X_val_seq, y_val_seq = self.create_sequences(X_val, y_val, seq_length)
        
        print(f"✓ Train sequences: {X_train_seq.shape}")
        print(f"✓ Val sequences: {X_val_seq.shape}")
        
        # Create DataLoaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_seq),
            torch.FloatTensor(y_train_seq)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val_seq),
            torch.FloatTensor(y_val_seq)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        print(f"\nTraining for {epochs} epochs...")
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), './forecasting/trained/lstm_best.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        print(f"\n✓ Training complete!")
        print(f"  Best val loss: {best_val_loss:.4f}")
        print("="*60)
        
        return self.train_losses, self.val_losses
    
    def predict(self, X, seq_length=24):
        """Make predictions"""
        self.model.eval()
        X_seq, _ = self.create_sequences(X, np.zeros(len(X)), seq_length)
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            preds = self.model(X_tensor).cpu().numpy().squeeze()
        
        return preds

if __name__ == "__main__":
    import os
    os.makedirs('./forecasting/trained', exist_ok=True)
    
    # Load data
    X_train = np.load('./data/processed/X_train.npy')
    y_train = np.load('./data/processed/y_train.npy')
    X_val = np.load('./data/processed/X_val.npy')
    y_val = np.load('./data/processed/y_val.npy')
    
    # Create and train model
    model = LSTMForecaster(input_size=X_train.shape[1], hidden_size=64)
    trainer = LSTMTrainer(model, lr=0.001)
    
    trainer.train(X_train, y_train, X_val, y_val, epochs=30, seq_length=24)
    
    # Evaluate
    y_pred = trainer.predict(X_val)
    y_true = y_val[24:]
    
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true)**2))
    mape = np.mean(np.abs((y_pred - y_true) / y_true)) * 100
    
    print(f"\n📊 LSTM Performance:")
    print(f"  MAE: {mae:.2f} kW")
    print(f"  RMSE: {rmse:.2f} kW")
    print(f"  MAPE: {mape:.2f}%")
