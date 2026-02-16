"""
Cognitive Smart Grid Dashboard
Interactive Streamlit dashboard for monitoring and control
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="Cognitive Smart Grid",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('./data/synthetic/smart_grid_data.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except:
        st.error("Could not load data. Please run data generation first.")
        return None

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">⚡ Cognitive Smart Grid Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📊 Controls")
    
    view = st.sidebar.selectbox(
        "Select View",
        ["Live Monitoring", "Forecasting", "Peak Detection", "Demand Response", "Analytics"]
    )
    
    # Load data
    df = load_data()
    
    if df is None:
        st.stop()
    
    # Tabs based on selection
    if view == "Live Monitoring":
        show_live_monitoring(df)
    elif view == "Forecasting":
        show_forecasting(df)
    elif view == "Peak Detection":
        show_peak_detection(df)
    elif view == "Demand Response":
        show_demand_response(df)
    else:
        show_analytics(df)

def show_live_monitoring(df):
    """Live monitoring view"""
    st.header("📈 Live Grid Monitoring")
    
    # Current metrics
    col1, col2, col3, col4 = st.columns(4)
    
    current_load = df.iloc[-1]['load_kw']
    current_price = df.iloc[-1]['price_per_kwh']
    current_carbon = df.iloc[-1]['carbon_intensity']
    current_renewable = df.iloc[-1]['renewable_pct']
    
    with col1:
        st.metric("Current Load", f"{current_load:.1f} kW", 
                 delta=f"{current_load - df.iloc[-2]['load_kw']:.1f}")
    with col2:
        st.metric("Price", f"${current_price:.3f}/kWh",
                 delta=f"${current_price - df.iloc[-2]['price_per_kwh']:.3f}")
    with col3:
        st.metric("Carbon", f"{current_carbon:.0f} gCO2/kWh",
                 delta=f"{current_carbon - df.iloc[-2]['carbon_intensity']:.0f}")
    with col4:
        st.metric("Renewable", f"{current_renewable:.1f}%",
                 delta=f"{current_renewable - df.iloc[-2]['renewable_pct']:.1f}")
    
    # Last 24 hours
    st.subheader("Last 24 Hours")
    
    last_24h = df.tail(96)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=last_24h['timestamp'],
        y=last_24h['load_kw'],
        mode='lines',
        name='Load',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.update_layout(
        title="Electricity Load",
        xaxis_title="Time",
        yaxis_title="Load (kW)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Price and Carbon
    col1, col2 = st.columns(2)
    
    with col1:
        fig_price = px.line(last_24h, x='timestamp', y='price_per_kwh',
                           title='Electricity Price')
        st.plotly_chart(fig_price, use_container_width=True)
    
    with col2:
        fig_carbon = px.line(last_24h, x='timestamp', y='carbon_intensity',
                            title='Carbon Intensity')
        st.plotly_chart(fig_carbon, use_container_width=True)

def show_forecasting(df):
    """Forecasting view"""
    st.header("🔮 Load Forecasting")
    
    st.write("**Next 24-Hour Forecast**")
    
    # Simple forecast
    last_week_same_time = df.tail(7 * 96)
    forecast_mean = last_week_same_time.groupby('hour')['load_kw'].mean()
    forecast_std = last_week_same_time.groupby('hour')['load_kw'].std()
    
    future_hours = pd.date_range(
        start=df.iloc[-1]['timestamp'] + timedelta(minutes=15),
        periods=96,
        freq='15min'
    )
    
    forecast_values = forecast_mean.values
    lower_bound = forecast_values - 1.96 * forecast_std.values
    upper_bound = forecast_values + 1.96 * forecast_std.values
    
    fig = go.Figure()
    
    # Historical
    last_24h = df.tail(96)
    fig.add_trace(go.Scatter(
        x=last_24h['timestamp'],
        y=last_24h['load_kw'],
        mode='lines',
        name='Historical',
        line=dict(color='blue')
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=future_hours,
        y=forecast_values,
        mode='lines',
        name='Forecast',
        line=dict(color='red', dash='dash')
    ))
    
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=list(future_hours) + list(future_hours[::-1]),
        y=list(upper_bound) + list(lower_bound[::-1]),
        fill='toself',
        fillcolor='rgba(255,0,0,0.1)',
        line=dict(color='rgba(255,0,0,0)'),
        name='95% CI',
        showlegend=True
    ))
    
    fig.update_layout(
        title="24-Hour Load Forecast",
        xaxis_title="Time",
        yaxis_title="Load (kW)",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Forecasted Peak", f"{forecast_values.max():.1f} kW")
    with col2:
        st.metric("Forecasted Average", f"{forecast_values.mean():.1f} kW")
    with col3:
        st.metric("Forecast Uncertainty", f"±{forecast_std.mean():.1f} kW")

def show_peak_detection(df):
    """Peak detection view"""
    st.header("⚠️ Peak Load Detection")
    
    # Calculate peak threshold
    peak_threshold = np.percentile(df['load_kw'], 95)
    
    # Current risk
    current_load = df.iloc[-1]['load_kw']
    risk_score = min(1.0, current_load / peak_threshold)
    
    if risk_score < 0.6:
        risk_level = "🟢 Low"
        risk_color = "green"
    elif risk_score < 0.8:
        risk_level = "🟡 Medium"
        risk_color = "yellow"
    elif risk_score < 0.95:
        risk_level = "🟠 High"
        risk_color = "orange"
    else:
        risk_level = "🔴 Critical"
        risk_color = "red"
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Risk Level", risk_level)
    with col2:
        st.metric("Risk Score", f"{risk_score:.2%}")
    with col3:
        st.metric("Peak Threshold", f"{peak_threshold:.1f} kW")
    
    # Risk history
    st.subheader("Risk History (Last 7 Days)")
    
    last_week = df.tail(7 * 96)
    risk_scores = last_week['load_kw'] / peak_threshold
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=last_week['timestamp'],
        y=risk_scores,
        mode='lines',
        fill='tozeroy',
        line=dict(color='orange'),
        name='Risk Score'
    ))
    
    fig.add_hline(y=0.95, line_dash="dash", line_color="red", 
                  annotation_text="Critical")
    fig.add_hline(y=0.8, line_dash="dash", line_color="orange",
                  annotation_text="High")
    fig.add_hline(y=0.6, line_dash="dash", line_color="yellow",
                  annotation_text="Medium")
    
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Risk Score",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_demand_response(df):
    """Demand response view"""
    st.header("🎯 Demand Response Optimization")
    
    st.write("**RL-Optimized Actions**")
    
    # Simulated DR actions
    actions = {
        "Reduce HVAC": {"reduction": 120, "cost_saving": 2.4, "carbon_saving": 50},
        "Delay EV": {"reduction": 90, "cost_saving": 1.8, "carbon_saving": 38},
        "Shift Appliances": {"reduction": 60, "cost_saving": 1.2, "carbon_saving": 25},
        "Activate Battery": {"reduction": 150, "cost_saving": 3.0, "carbon_saving": 63}
    }
    
    current_load = df.iloc[-1]['load_kw']
    
    # Recommendation
    recommended = "Reduce HVAC" if current_load > 800 else "Shift Appliances"
    
    st.info(f"**Recommended Action:** {recommended}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Load Reduction", f"{actions[recommended]['reduction']} kW")
    with col2:
        st.metric("Cost Savings", f"${actions[recommended]['cost_saving']:.2f}")
    with col3:
        st.metric("Carbon Savings", f"{actions[recommended]['carbon_saving']} gCO2")
    
    # All actions
    st.subheader("Available Actions")
    
    action_df = pd.DataFrame(actions).T
    action_df.index.name = "Action"
    
    fig = px.bar(action_df, y=['reduction', 'carbon_saving'],
                title='Action Comparison',
                barmode='group')
    st.plotly_chart(fig, use_container_width=True)

def show_analytics(df):
    """Analytics view"""
    st.header("📊 System Analytics")
    
    # Overall stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", f"{len(df):,}")
    with col2:
        st.metric("Avg Load", f"{df['load_kw'].mean():.1f} kW")
    with col3:
        st.metric("Peak Load", f"{df['load_kw'].max():.1f} kW")
    with col4:
        st.metric("Avg Carbon", f"{df['carbon_intensity'].mean():.0f} gCO2/kWh")
    
    # Load distribution
    st.subheader("Load Distribution")
    
    fig = px.histogram(df, x='load_kw', nbins=50,
                      title='Load Distribution')
    st.plotly_chart(fig, use_container_width=True)
    
    # Hourly patterns
    st.subheader("Average Daily Pattern")
    
    hourly_avg = df.groupby('hour').agg({
        'load_kw': 'mean',
        'price_per_kwh': 'mean',
        'carbon_intensity': 'mean'
    }).reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hourly_avg['hour'],
        y=hourly_avg['load_kw'],
        mode='lines+markers',
        name='Load'
    ))
    
    fig.update_layout(
        xaxis_title="Hour of Day",
        yaxis_title="Average Load (kW)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("Feature Correlations")
    
    corr_df = df[['load_kw', 'temperature_c', 'humidity_pct', 
                  'price_per_kwh', 'carbon_intensity']].corr()
    
    fig = px.imshow(corr_df, text_auto=True, aspect="auto",
                    color_continuous_scale='RdBu_r')
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
