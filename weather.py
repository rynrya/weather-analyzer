import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
import google.generativeai as genai

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Climate AI Forecaster", layout="wide")
st.title("Next-Gen Temperature Forecaster 🌤️")

st.sidebar.header("Search Parameters 🔍")
st.sidebar.write("Input any global location to generate a custom 2027 forecast model.")

user_target_city = st.sidebar.text_input("Target Location:", "New York")

# --- 2. GPS LOOKUP FUNCTION ---
def fetch_gps_data(location_string):
    """Hits the geocoding API to find exact latitude and longitude."""
    api_endpoint = f"https://geocoding-api.open-meteo.com/v1/search?name={location_string}&count=1&language=en&format=json"
    api_reply = requests.get(api_endpoint).json()
    
    if "results" in api_reply:
        target_lat = api_reply["results"][0]["latitude"]
        target_lon = api_reply["results"][0]["longitude"]
        target_country = api_reply["results"][0].get("country", "")
        formatted_name = f"{api_reply['results'][0]['name']}, {target_country}"
        return target_lat, target_lon, formatted_name
    return None, None, None

latitude, longitude, official_city_name = fetch_gps_data(user_target_city)

# --- 3. CORE LOGIC & DASHBOARD ---
if latitude is None:
    st.error(f"Error: Unable to locate '{user_target_city}'. Please verify the spelling.")
else:
    st.sidebar.success(f"Successfully locked onto: {official_city_name}")

    # Data Retrieval caching
    @st.cache_data
    def pull_historical_climate(lat, lon):
        weather_url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date=2016-01-01&end_date=2026-03-31&daily=temperature_2m_max&timezone=auto"
        raw_json = requests.get(weather_url).json()
        climate_table = pd.DataFrame(raw_json["daily"])
        
        climate_table['time'] = pd.to_datetime(climate_table['time'])
        climate_table['Year'] = climate_table['time'].dt.year
        climate_table['Month'] = climate_table['time'].dt.month
        climate_table['Day'] = climate_table['time'].dt.day
        return climate_table

    historical_df = pull_historical_climate(latitude, longitude)

    # --- 4. MACHINE LEARNING ENGINE ---
    st.subheader(f"📊 2027 Predictive Model: {official_city_name}")

    features = historical_df[['Year', 'Month', 'Day']] 
    target = historical_df['temperature_2m_max']
    
    rf_engine = RandomForestRegressor(n_estimators=18, max_depth=12, n_jobs=-1, random_state=99)
    rf_engine.fit(features, target) 

    upcoming_days = pd.date_range(start='2027-01-01', end='2027-12-31')
    prediction_calendar = pd.DataFrame({
        'Year': upcoming_days.year, 
        'Month': upcoming_days.month, 
        'Day': upcoming_days.day
    })
    
    forecasted_temps = rf_engine.predict(prediction_calendar)

    # --- 5. INTERACTIVE PLOTLY VISUALIZATION ---
    past_records = pd.DataFrame({'Timeline': historical_df['time'], 'Heat Level': target, 'Dataset': 'Archived Data'})
    future_records = pd.DataFrame({'Timeline': upcoming_days, 'Heat Level': forecasted_temps, 'Dataset': 'AI Prediction'})
    master_chart_data = pd.concat([past_records, future_records])

    interactive_chart = px.line(
        master_chart_data, 
        x='Timeline', 
        y='Heat Level', 
        color='Dataset',
        title=f"Historical Climate vs 2027 Forecast ({official_city_name})",
        color_discrete_sequence=['#555555', '#ffaa00'] 
    )
    
    st.plotly_chart(interactive_chart, use_container_width=True)

    st.divider()

    # --- 6. CUSTOM AI DATA AGENT ---
    st.subheader("🧠 Chat with the Weather AI")
    user_query = st.text_input("Ask about the data (e.g., 'What was the hottest day?', 'What is the average temperature?'):")

    if user_query:
        with st.spinner("The AI is analyzing the data cheat sheet..."):
            try:
                # 1. Authenticate with Gemini
                genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
                
                # 2. Build the "Cheat Sheet" for the AI
                df_summary = historical_df['temperature_2m_max'].describe().to_string()
                hottest_idx = historical_df['temperature_2m_max'].idxmax()
                coldest_idx = historical_df['temperature_2m_max'].idxmin()
                
                hottest_day = historical_df.loc[hottest_idx]
                coldest_day = historical_df.loc[coldest_idx]
                
                # 3. Write the advanced instructions for the AI
                system_prompt = f"""
                You are an Expert Climate Data Scientist and Analytical Engine assigned to {official_city_name}. 
                You have been provided with a raw, daily dataset of maximum temperatures (°C) from 2016 to 2026.

                RAW DATA EXPORT:
                {csv_data}

                YOUR ANALYTICAL DIRECTIVES:
                1. STRICT ACCURACY: When a user asks for an average, minimum, or maximum for a specific timeframe (e.g., "March 2023"), you must locate EVERY row matching that "YYYY-MM" timeframe in the raw data, extract the values, and calculate the exact math. 
                2. ENHANCED METRICS: Do not just give a single number. Enhance your response by providing context. For example, if asked for an average, also mention the absolute high and low for that specific period, or how it compares to the overall 10-year baseline.
                3. TREND IDENTIFICATION: If the user asks about trends, look at year-over-year or month-over-month data to calculate the percentage change or temperature delta.
                4. DATA INTEGRITY: Base your calculations ONLY on the provided CSV data. If a specific date or month is missing from the raw data, explicitly state that it is unavailable. Do not estimate or hallucinate missing values.
                
                Format your final answer to be clear, highly professional, and easy to read for a dashboard user.
                
                USER QUERY: {user_query}
                """
                
                # 4. Generate the response
                model = genai.GenerativeModel('gemini-2.5-flash')
                ai_answer = model.generate_content(system_prompt)
                
                st.success(ai_answer.text)
            except Exception as e:
                st.error(f"Something went wrong with the AI connection: {e}")