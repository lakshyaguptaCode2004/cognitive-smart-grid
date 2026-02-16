"""
FastAPI Backend for Cognitive Smart Grid
Provides REST API for forecasting, peak detection, and demand response
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import numpy as np
import pandas as pd
import joblib
import torch
from datetime import datetime

app = FastAPI(title="Cognitive Smart Grid API", version="1.0.0")

# Load data and models at startup
try:
    df = pd.read_csv('./data/synthetic/smart_grid_data.csv')
    scaler = joblib.load('./data/processed/scaler.pkl')
    print("✓ Data and models loaded")
except Exception as e:
    print(f"⚠ Warning: Could not load models: {e}")
    df = None
    scaler = None

# Pydantic models
class ForecastRequest(BaseModel):
    hours_ahead: int = 24

class ForecastResponse(BaseModel):
    timestamp: str
    forecasted_load: List[float]
    confidence_lower: List[float]
    confidence_upper: List[float]

class PeakRiskResponse(BaseModel):
    timestamp: str
    risk_score: float
    risk_level: str
    peak_probability: float

class OptimizationRequest(BaseModel):
    current_load: float
    price: float
    carbon: float
    user_comfort: float = 0.5

class OptimizationResponse(BaseModel):
    recommended_action: str
    expected_reduction: float
    cost_savings: float
    carbon_savings: float

# Routes
@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "Cognitive Smart Grid API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": [
            "/forecast",
            "/peak-risk",
            "/optimize",
            "/health"
        ]
    }

@app.get("/health")
def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "data_loaded": df is not None,
        "models_loaded": scaler is not None
    }

@app.post("/forecast", response_model=ForecastResponse)
def forecast_load(request: ForecastRequest):
    """Forecast electricity load"""
    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    # Simple forecast: use historical pattern + noise
    hours_ahead = request.hours_ahead
    current_idx = len(df) - 1
    
    # Get similar historical pattern
    current_hour = df.iloc[current_idx]['hour']
    current_dow = df.iloc[current_idx]['day_of_week']
    
    # Find similar times
    similar_mask = (df['hour'] == current_hour) & (df['day_of_week'] == current_dow)
    similar_loads = df[similar_mask]['load_kw'].values
    
    if len(similar_loads) > 0:
        base_forecast = np.mean(similar_loads)
        std_forecast = np.std(similar_loads)
    else:
        base_forecast = df['load_kw'].mean()
        std_forecast = df['load_kw'].std()
    
    # Generate forecast
    forecasted_load = [base_forecast + np.random.normal(0, std_forecast * 0.1) 
                      for _ in range(hours_ahead)]
    
    confidence_lower = [f - 1.96 * std_forecast for f in forecasted_load]
    confidence_upper = [f + 1.96 * std_forecast for f in forecasted_load]
    
    return ForecastResponse(
        timestamp=datetime.now().isoformat(),
        forecasted_load=forecasted_load,
        confidence_lower=confidence_lower,
        confidence_upper=confidence_upper
    )

@app.get("/peak-risk", response_model=PeakRiskResponse)
def get_peak_risk():
    """Get current peak load risk"""
    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    # Calculate peak threshold
    peak_threshold = np.percentile(df['load_kw'], 95)
    
    # Current load (last value)
    current_load = df.iloc[-1]['load_kw']
    
    # Risk score
    risk_score = min(1.0, current_load / peak_threshold)
    
    # Peak probability (simple Gaussian model)
    mean_load = df['load_kw'].mean()
    std_load = df['load_kw'].std()
    z_score = (current_load - mean_load) / std_load
    peak_prob = min(1.0, max(0.0, (z_score - 1) / 2))
    
    # Risk level
    if risk_score < 0.6:
        risk_level = "low"
    elif risk_score < 0.8:
        risk_level = "medium"
    elif risk_score < 0.95:
        risk_level = "high"
    else:
        risk_level = "critical"
    
    return PeakRiskResponse(
        timestamp=datetime.now().isoformat(),
        risk_score=round(risk_score, 3),
        risk_level=risk_level,
        peak_probability=round(peak_prob, 3)
    )

@app.post("/optimize", response_model=OptimizationResponse)
def optimize_demand_response(request: OptimizationRequest):
    """Optimize demand response action"""
    
    # Simple rule-based optimization (would use RL in full system)
    actions = {
        0: ("reduce_hvac_20%", 0.20),
        1: ("delay_ev_charging", 0.15),
        2: ("shift_appliances", 0.10),
        3: ("activate_battery", 0.25),
        4: ("no_action", 0.00)
    }
    
    # Decision logic
    if request.current_load > 800 and request.price > 0.15:
        action_idx = 0  # Reduce HVAC
    elif request.carbon > 500 and request.user_comfort > 0.6:
        action_idx = 3  # Use battery
    elif request.price > 0.18:
        action_idx = 1  # Delay EV
    elif request.current_load > 700:
        action_idx = 2  # Shift appliances
    else:
        action_idx = 4  # No action
    
    action_name, reduction_pct = actions[action_idx]
    
    expected_reduction = request.current_load * reduction_pct
    cost_savings = expected_reduction * request.price
    carbon_savings = expected_reduction * request.carbon / 1000
    
    return OptimizationResponse(
        recommended_action=action_name,
        expected_reduction=round(expected_reduction, 2),
        cost_savings=round(cost_savings, 2),
        carbon_savings=round(carbon_savings, 2)
    )

@app.get("/stats")
def get_statistics():
    """Get dataset statistics"""
    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    return {
        "total_samples": len(df),
        "load_stats": {
            "mean": round(df['load_kw'].mean(), 2),
            "std": round(df['load_kw'].std(), 2),
            "min": round(df['load_kw'].min(), 2),
            "max": round(df['load_kw'].max(), 2)
        },
        "price_stats": {
            "mean": round(df['price_per_kwh'].mean(), 3),
            "min": round(df['price_per_kwh'].min(), 3),
            "max": round(df['price_per_kwh'].max(), 3)
        },
        "carbon_stats": {
            "mean": round(df['carbon_intensity'].mean(), 1),
            "min": round(df['carbon_intensity'].min(), 1),
            "max": round(df['carbon_intensity'].max(), 1)
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting Cognitive Smart Grid API...")
    print("📍 Access API at: http://localhost:8000")
    print("📚 Documentation at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
