import requests
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

@st.cache_data
def fetch_gps_data(location_string):
    try:
        api_endpoint = f"https://geocoding-api.open-meteo.com/v1/search?name={location_string}&count=1&language=en&format=json"
        api_reply = requests.get(api_endpoint, timeout=10).json()
        if "results" in api_reply:
            lat = api_reply["results"][0]["latitude"]
            lon = api_reply["results"][0]["longitude"]
            country = api_reply["results"][0].get("country", "")
            return lat, lon, f"{api_reply['results'][0]['name']}, {country}"
    except Exception as e:
        return None, None, f"Error: {e}"
    return None, None, None

@st.cache_data
def pull_historical_climate(lat, lon):
    weather_url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date=2016-01-01&end_date=2026-03-31&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max&timezone=auto"
    raw_json = requests.get(weather_url).json()
    df = pd.DataFrame(raw_json["daily"]).dropna()
    df['time'] = pd.to_datetime(df['time'])
    df['Year'] = df['time'].dt.year
    df['Month'] = df['time'].dt.month
    df['Day'] = df['time'].dt.day
    return df

@st.cache_data
def generate_2027_forecast(historical_df):
    features = historical_df[['Year', 'Month', 'Day']] 
    targets = historical_df[['temperature_2m_max', 'temperature_2m_min', 'precipitation_sum', 'wind_speed_10m_max']]
    
    rf = RandomForestRegressor(n_estimators=25, max_depth=15, n_jobs=-1, random_state=42)
    rf.fit(features, targets) 

    upcoming_days = pd.date_range(start='2027-01-01', end='2027-12-31')
    pred_cal = pd.DataFrame({'Year': upcoming_days.year, 'Month': upcoming_days.month, 'Day': upcoming_days.day})
    
    forecasted_data = rf.predict(pred_cal)
    future_df = pd.DataFrame(forecasted_data, columns=targets.columns)
    future_df['time'] = upcoming_days
    return future_df