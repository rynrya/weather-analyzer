import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_manager import fetch_gps_data, pull_historical_climate, generate_2027_forecast

st.set_page_config(page_title="Data Dashboard", page_icon="📊", layout="wide")

city = st.session_state.get('city_input', 'Nepal')
lat, lon, official_name = fetch_gps_data(city)

if lat is None:
    st.error("Please set a valid location on the Home page.")
    st.stop()

st.title(f"📊 Meteorological Dashboard: {official_name}")

with st.spinner("Downloading and processing data..."):
    historical_df = pull_historical_climate(lat, lon)
    future_df = generate_2027_forecast(historical_df)

# KPI Metrics
st.subheader("10-Year Historical Averages")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg Max Temp", f"{historical_df['temperature_2m_max'].mean():.1f} °C")
col2.metric("Avg Min Temp", f"{historical_df['temperature_2m_min'].mean():.1f} °C")
col3.metric("Daily Avg Rain", f"{historical_df['precipitation_sum'].mean():.2f} mm")
col4.metric("Avg Max Wind", f"{historical_df['wind_speed_10m_max'].mean():.1f} km/h")

st.divider()

# Charts
st.subheader("📈 Interactive Climate Projection")
display_options = {
    'temperature_2m_max': 'Max Temperature (°C)',
    'temperature_2m_min': 'Min Temperature (°C)',
    'precipitation_sum': 'Precipitation (mm)',
    'wind_speed_10m_max': 'Max Wind Speed (km/h)'
}
selected_metric = st.selectbox("Select a metric:", list(display_options.keys()), format_func=lambda x: display_options[x])

past_records = pd.DataFrame({'Timeline': historical_df['time'], 'Value': historical_df[selected_metric], 'Dataset': 'Archived Data'})
future_records = pd.DataFrame({'Timeline': future_df['time'], 'Value': future_df[selected_metric], 'Dataset': 'AI Prediction (2027)'})

interactive_chart = px.line(
    pd.concat([past_records, future_records]), 
    x='Timeline', y='Value', color='Dataset',
    color_discrete_sequence=['#555555', '#00cc96'] 
)
st.plotly_chart(interactive_chart, use_container_width=True)