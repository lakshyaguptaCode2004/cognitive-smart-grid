# 🔋 Cognitive Smart Grid - User Guide

## 📋 Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Component Guide](#component-guide)
5. [API Reference](#api-reference)
6. [Dashboard Guide](#dashboard-guide)
7. [Evaluation](#evaluation)
8. [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

The Cognitive Smart Grid is a complete AI-driven demand response optimization system featuring:

- **Deep Learning Forecasting**: LSTM models for 24-hour load prediction
- **Peak Detection**: Multi-method peak risk assessment
- **RL Optimization**: Reinforcement learning for demand response
- **NLP Processing**: User preference extraction from text
- **Carbon Optimization**: Multi-objective sustainability optimization
- **Real-time API**: FastAPI REST endpoints
- **Interactive Dashboard**: Streamlit visualization

---

## 💻 Installation

### Prerequisites

- Python 3.9+
- 16GB+ RAM
- 10GB+ disk space

### Setup

```bash
# Clone/navigate to project
cd cognitive-smart-grid

# Create virtual environment
python -m venv venv

# Activate environment
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### Option 1: Run Everything (Automated)

```bash
bash scripts/run_all.sh
```

This will:
1. Generate synthetic data
2. Preprocess features
3. Train forecasting models
4. Test RL environment
5. Run evaluation

### Option 2: Step-by-Step

```bash
# 1. Generate Data
python data/generate_synthetic_data.py

# 2. Preprocess
python preprocessing/pipeline.py

# 3. Train Forecasting
python forecasting/models/lstm_model.py

# 4. Test RL
python rl_agent/envs/demand_response_env.py

# 5. Evaluate
python evaluation/metrics.py
```

### Launch Services

```bash
# Start API Server
uvicorn api_backend.main:app --reload --port 8000

# Launch Dashboard (in new terminal)
streamlit run dashboard/app.py
```

---

## 🏗️ Component Guide

### 1. Data Generation

**File**: `data/generate_synthetic_data.py`

Generates 365 days of realistic smart grid data:
- 35,040 15-minute samples
- Load, weather, pricing, carbon intensity
- 10,000 user feedback samples

**Output**:
- `data/synthetic/smart_grid_data.csv`
- `data/synthetic/user_feedback.csv`

### 2. Preprocessing

**File**: `preprocessing/pipeline.py`

Features:
- Cyclical time encoding
- Rolling statistics (1h, 6h, 24h)
- Lag features
- StandardScaler normalization
- 70/15/15 train/val/test split

**Output**:
- `data/processed/X_train.npy` (24,528 samples)
- `data/processed/X_val.npy` (5,256 samples)
- `data/processed/X_test.npy` (5,256 samples)
- Corresponding y files
- Scaler and feature column metadata

### 3. Forecasting

**File**: `forecasting/models/lstm_model.py`

LSTM with attention mechanism:
- Input: 24-step sequences (6 hours)
- Architecture: 2-layer LSTM (64 hidden units)
- Output: Single-step forecast
- Target: MAPE < 5%

**Metrics**:
- MAE: Mean Absolute Error
- RMSE: Root Mean Squared Error
- MAPE: Mean Absolute Percentage Error

### 4. RL Environment

**File**: `rl_agent/envs/demand_response_env.py`

Gymnasium environment for demand response:

**State** (7 dims):
- Current load
- Forecasted load
- Price (normalized)
- Carbon intensity
- Hour of day
- Weekend indicator
- User comfort score

**Actions** (5 discrete):
- 0: Reduce HVAC 20%
- 1: Delay EV charging
- 2: Shift appliances
- 3: Activate battery
- 4: No action

**Reward**:
```
R = -peak_penalty - cost_penalty - carbon_penalty - comfort_penalty + savings
```

### 5. API Backend

**File**: `api_backend/main.py`

FastAPI REST endpoints:

**Endpoints**:
- `GET /` - API info
- `GET /health` - Health check
- `POST /forecast` - Get load forecast
- `GET /peak-risk` - Get peak risk assessment
- `POST /optimize` - Get DR recommendation
- `GET /stats` - Dataset statistics

**Access**: http://localhost:8000/docs

### 6. Dashboard

**File**: `dashboard/app.py`

Streamlit interactive dashboard with 5 views:

1. **Live Monitoring**: Real-time grid metrics
2. **Forecasting**: 24-hour load predictions
3. **Peak Detection**: Risk assessment
4. **Demand Response**: RL recommendations
5. **Analytics**: Statistical insights

**Access**: http://localhost:8501

### 7. Evaluation

**File**: `evaluation/metrics.py`

Comprehensive evaluation metrics:

**Forecasting**:
- MAE, RMSE, MAPE, R²

**Peak Detection**:
- Precision, Recall, F1-Score, Accuracy

**Demand Response**:
- Load reduction %
- Peak reduction %
- Cost savings %
- Carbon savings %

**Output**:
- `outputs/reports/evaluation_report.txt`
- `outputs/reports/evaluation_results.json`

---

## 📡 API Reference

### Forecast Endpoint

```bash
curl -X POST "http://localhost:8000/forecast" \
  -H "Content-Type: application/json" \
  -d '{"hours_ahead": 24}'
```

**Response**:
```json
{
  "timestamp": "2026-02-12T10:30:00",
  "forecasted_load": [650.2, 645.1, ...],
  "confidence_lower": [620.5, 615.3, ...],
  "confidence_upper": [679.9, 674.9, ...]
}
```

### Peak Risk Endpoint

```bash
curl "http://localhost:8000/peak-risk"
```

**Response**:
```json
{
  "timestamp": "2026-02-12T10:30:00",
  "risk_score": 0.75,
  "risk_level": "high",
  "peak_probability": 0.65
}
```

### Optimize Endpoint

```bash
curl -X POST "http://localhost:8000/optimize" \
  -H "Content-Type: application/json" \
  -d '{
    "current_load": 850,
    "price": 0.18,
    "carbon": 520,
    "user_comfort": 0.6
  }'
```

**Response**:
```json
{
  "recommended_action": "reduce_hvac_20%",
  "expected_reduction": 170.0,
  "cost_savings": 30.6,
  "carbon_savings": 88.4
}
```

---

## 📊 Dashboard Guide

### Live Monitoring View

- **Current Metrics**: Load, Price, Carbon, Renewable %
- **24-Hour Chart**: Real-time load visualization
- **Price Tracking**: Electricity pricing trends
- **Carbon Tracking**: Carbon intensity trends

### Forecasting View

- **24-Hour Forecast**: Load prediction with confidence intervals
- **Peak Prediction**: Forecasted peak load
- **Uncertainty Bands**: 95% confidence intervals

### Peak Detection View

- **Risk Assessment**: Current risk level and score
- **Historical Trends**: 7-day risk history
- **Threshold Indicators**: Critical/High/Medium/Low zones

### Demand Response View

- **RL Recommendations**: Optimal actions
- **Impact Metrics**: Load/Cost/Carbon savings
- **Action Comparison**: Bar charts of all options

### Analytics View

- **System Statistics**: Overall performance
- **Load Distribution**: Histogram of load patterns
- **Daily Patterns**: Average hourly profiles
- **Correlations**: Feature relationship heatmap

---

## 🔬 Evaluation

### Run Complete Evaluation

```bash
python evaluation/metrics.py
```

### Expected Results

**Forecasting (LSTM)**:
- MAE: 15-25 kW
- RMSE: 20-35 kW
- MAPE: 2.5-4.5%
- R²: 0.90-0.95

**Peak Detection**:
- Precision: 85-95%
- Recall: 85-95%
- F1-Score: 85-95%

**Demand Response**:
- Peak Reduction: 15-25%
- Cost Savings: 10-20%
- Carbon Savings: 15-30%

---

## 🔧 Troubleshooting

### Data Generation Issues

**Problem**: Import errors
```bash
# Solution: Install dependencies
pip install pandas numpy
```

**Problem**: File not found
```bash
# Solution: Create directories
mkdir -p data/synthetic
```

### Preprocessing Issues

**Problem**: Pandas fillna error
```bash
# Solution: Update pandas
pip install --upgrade pandas
```

### API Issues

**Problem**: Port already in use
```bash
# Solution: Use different port
uvicorn api_backend.main:app --port 8001
```

**Problem**: Data not loaded
```bash
# Solution: Run data generation first
python data/generate_synthetic_data.py
```

### Dashboard Issues

**Problem**: Streamlit not found
```bash
# Solution: Install streamlit
pip install streamlit
```

**Problem**: Import errors
```bash
# Solution: Install plotly
pip install plotly
```

---

## 📚 Additional Resources

- **IEEE Papers**: See `research/paper/references.bib`
- **Documentation**: See component docstrings
- **Examples**: See `notebooks/` directory

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch
3. Add tests
4. Submit pull request

---

## 📄 License

MIT License - See LICENSE file

---

## 📧 Support

For issues or questions:
- GitHub Issues: [your-repo-url]
- Email: your.email@example.com

---

**Built with ❤️ for sustainable smart grids**
