import streamlit as st
import google.generativeai as genai
from utils.data_manager import fetch_gps_data, pull_historical_climate, generate_2027_forecast

st.set_page_config(page_title="AI Chat", page_icon="🧠", layout="wide")

city = st.session_state.get('city_input', 'Nepal')
lat, lon, official_name = fetch_gps_data(city)

if lat is None:
    st.error("Please set a valid location on the Home page.")
    st.stop()

st.title(f"🧠 Meteorological AI: {official_name}")
st.markdown("Ask anything about the 10-year historical data or the 2027 ML predictions.")

user_query = st.text_input("Enter your query here:")

if user_query:
    with st.spinner("Gemini is analyzing the data..."):
        try:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            
            # Fetch data quietly
            historical_df = pull_historical_climate(lat, lon)
            future_df = generate_2027_forecast(historical_df)
            
            cols = ['time', 'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum', 'wind_speed_10m_max']
            historical_csv = historical_df[cols].to_csv(index=False)
            prediction_csv = future_df[cols].to_csv(index=False)
            
            system_prompt = f"""
            You are an Expert Meteorologist assigned to {official_name} (Lat: {lat}, Lon: {lon}).
            
            HISTORICAL DATA (2016-2026):
            {historical_csv}
            
            PREDICTED DATA (2027):
            {prediction_csv}
            
            DIRECTIVES: Use exact math based on the YYYY-MM dates provided. Contrast past data with 2027 predictions when relevant. Provide context (max/mins) when giving averages. Base answers ONLY on this data.
            
            USER QUERY: {user_query}
            """
            
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = model.generate_content(system_prompt)
            
            st.success(response.text)
        except Exception as e:
            st.error(f"Error: {e}")