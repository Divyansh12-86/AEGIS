import streamlit as st
import requests
from dashboard.components.strategy_table import render_strategy_table
import os
from dotenv import load_dotenv

load_dotenv()

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="AEGIS — Strategy Explorer", layout="wide")
st.title("Strategy Explorer")

assets = requests.get(f"{API_BASE}/telemetry/assets").json()
asset_ids = [a["asset_id"] for a in assets]

if not asset_ids:
    st.info("No telemetry yet. Start the Modbus simulator to populate asset data.")
else:
    selected = st.selectbox("Select Asset", asset_ids)

    if st.button("Generate Strategies", type="primary"):
        with st.spinner("Generating and validating strategies..."):
            plans = requests.get(f"{API_BASE}/plans/{selected}").json()

        st.metric("Health Index", f"{plans['health_index']:.1f}%")
        render_strategy_table(plans["strategies"])
