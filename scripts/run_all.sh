#!/bin/bash

# Cognitive Smart Grid - Master Run Script
# Executes complete pipeline from data generation to evaluation

echo "========================================================================"
echo "COGNITIVE SMART GRID - COMPLETE PIPELINE"
echo "========================================================================"
echo ""

# Step 1: Generate Data
echo "Step 1/5: Generating synthetic data..."
python data/generate_synthetic_data.py
if [ $? -ne 0 ]; then
    echo "❌ Data generation failed"
    exit 1
fi
echo "✓ Data generation complete"
echo ""

# Step 2: Preprocess
echo "Step 2/5: Preprocessing data..."
python preprocessing/pipeline.py
if [ $? -ne 0 ]; then
    echo "❌ Preprocessing failed"
    exit 1
fi
echo "✓ Preprocessing complete"
echo ""

# Step 3: Train Forecasting (if LSTM exists)
echo "Step 3/5: Training forecasting models..."
if [ -f "forecasting/models/lstm_model.py" ]; then
    python forecasting/models/lstm_model.py
    echo "✓ Forecasting training complete"
else
    echo "⚠ Forecasting model not found, skipping..."
fi
echo ""

# Step 4: Test RL Environment
echo "Step 4/5: Testing RL environment..."
if [ -f "rl_agent/envs/demand_response_env.py" ]; then
    python rl_agent/envs/demand_response_env.py
    echo "✓ RL environment test complete"
else
    echo "⚠ RL environment not found, skipping..."
fi
echo ""

# Step 5: Run Evaluation
echo "Step 5/5: Running evaluation..."
python evaluation/metrics.py
if [ $? -ne 0 ]; then
    echo "❌ Evaluation failed"
    exit 1
fi
echo "✓ Evaluation complete"
echo ""

echo "========================================================================"
echo "PIPELINE COMPLETE"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "  1. Start API: uvicorn api_backend.main:app --reload"
echo "  2. Launch Dashboard: streamlit run dashboard/app.py"
echo ""
