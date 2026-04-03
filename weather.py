import streamlit as st
import requests
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import plotly.express as px


# 1. SET UP THE WEB PAGE
st.set_page_config(page_title="Weather AI", layout="wide")
st.title("🌡️ AI Weather Predictor & Analyzer")

st.sidebar.header("🌍 World Tour Search")
st.sidebar.write("Type any city on Earth to analyze its history and predict 2024!")

# --- THE NEW GLOBAL SEARCH ENGINE ---
# This creates a blank search bar for you to type in
city_query = st.sidebar.text_input("Enter a city name:", "Delhi")

# The digital phone book function
def get_coordinates(city_name):
    # This API translates city names into GPS coordinates
    geocode_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1&language=en&format=json"
    response = requests.get(geocode_url).json()
    
    # If the API successfully finds the city on the map
    if "results" in response:
        lat = response["results"][0]["latitude"]
        lon = response["results"][0]["longitude"]
        country = response["results"][0].get("country", "")
        full_name = f"{response['results'][0]['name']}, {country}"
        return lat, lon, full_name
    else:
        return None, None, None

# Run the search!
lat, lon, exact_name = get_coordinates(city_query)

# If the user types a typo or a fake city, we show an error instead of crashing
if lat is None:
    st.error(f"Could not find coordinates for '{city_query}'. Please check your spelling!")
else:
    # Tell the user exactly what location we locked onto
    st.sidebar.success(f"Located: {exact_name} (Lat: {lat:.2f}, Lon: {lon:.2f})")

    # 2. CACHE THE DATA
    @st.cache_data
    def load_data(latitude, longitude):
# Notice the end_date is now updated to pull data up to early 2026
        weather_url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date=2016-01-01&end_date=2026-03-31&daily=temperature_2m_max&timezone=auto"        
        data = requests.get(weather_url).json()
        df = pd.DataFrame(data["daily"])
        df['time'] = pd.to_datetime(df['time'])
        df['Year'] = df['time'].dt.year
        df['Month'] = df['time'].dt.month
        df['Day'] = df['time'].dt.day
        return df

    # We feed the dynamic GPS coordinates to our AI
    df = load_data(lat, lon)

    # 3. INTERACTIVE Q&A SECTION
    st.subheader(f"💬 Ask the Data about {exact_name}")
    question = st.text_input("Type your question here (e.g., 'What was the hottest day?', 'Coldest day?'):").lower()

    if "hottest" in question:
        hottest_row = df.loc[df['temperature_2m_max'].idxmax()]
        st.success(f"🔥 The absolute hottest day in {exact_name} was {hottest_row['time'].date()} at {hottest_row['temperature_2m_max']}°C.")
    elif "coldest" in question:
        coldest_row = df.loc[df['temperature_2m_max'].idxmin()]
        st.info(f"❄️ The absolute coldest day in {exact_name} was {coldest_row['time'].date()} at {coldest_row['temperature_2m_max']}°C.")
    elif question != "":
        st.warning("I'm still learning! Try asking about the 'hottest' or 'coldest' day.")

    st.divider() 

    # 4. THE AI & GRAPH SECTION
    st.subheader(f"📈 2024 AI Temperature Forecast for {exact_name}")

    X = df[['Year', 'Month', 'Day']] 
    y = df['temperature_2m_max']
    model = RandomForestRegressor(n_estimators=15, max_depth=10, n_jobs=-1, random_state=42)
    model.fit(X, y) 

    future_dates = pd.date_range(start='2027-01-01', end='2027-12-31')
    future_df = pd.DataFrame({'Year': future_dates.year, 'Month': future_dates.month, 'Day': future_dates.day})
    predictions_2024 = model.predict(future_df)


    # --- THE INTERACTIVE GRAPH UPGRADE ---
    # We package our past history and future predictions into one clean format
    history_df = pd.DataFrame({'Date': df['time'], 'Temperature': y, 'Type': 'Actual History'})
    future_df_plot = pd.DataFrame({'Date': future_dates, 'Temperature': predictions_2024, 'Type': '2024 AI Forecast'})
    combined_df = pd.concat([history_df, future_df_plot])

    # Plotly draws a dynamic, hoverable graph instantly
    fig = px.line(combined_df, x='Date', y='Temperature', color='Type',
                title=f"{exact_name} - Interactive Temperature Forecast",
                color_discrete_sequence=['#888888', '#00ff00']) # Gray for past, Neon Green for future

    # Display it perfectly fitted to your Streamlit webpage
    st.plotly_chart(fig, use_container_width=True)