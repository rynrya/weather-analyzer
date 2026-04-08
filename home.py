import streamlit as st
from utils.data_manager import fetch_gps_data

st.set_page_config(page_title="Ultimate Climate AI", page_icon="🌍", layout="wide")

st.title("🌍 Welcome to the Climate Forecaster")
st.markdown("Use the sidebar to set your target location, then navigate to the **Dashboard** or **AI Chat** pages to view the analysis.")

# Use Session State to remember the city across all pages
if 'city_input' not in st.session_state:
    st.session_state['city_input'] = "Nepal"

new_city = st.sidebar.text_input("Set Target Location:", st.session_state['city_input'])

if new_city != st.session_state['city_input']:
    st.session_state['city_input'] = new_city
    st.rerun()

lat, lon, official_name = fetch_gps_data(st.session_state['city_input'])

if lat is not None:
    st.success(f"📍 GPS Locked onto: **{official_name}**")
    st.info("👈 Please select a page from the sidebar to begin your analysis.")
else:
    st.error("Could not find that location. Please try another city.")