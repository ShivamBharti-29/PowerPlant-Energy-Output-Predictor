import streamlit as st
from prediction import predict_power
import joblib
import numpy as np

st.title("⚡ Power Plant Energy Prediction")

# Using columns for a cleaner UI
col1, col2 = st.columns(2)

with col1:
    AT = st.number_input("Temperature (AT)", value=15.0)
    V = st.number_input("Exhaust Vacuum (V)", value=40.0)
with col2:
    AP = st.number_input("Ambient Pressure (AP)", value=1000.0)
    RH = st.number_input("Relative Humidity (RH)", value=70.0)

if st.button("Predict"):
    # 1. Wrap data in a numpy array
    data = np.array([[AT, V, AP, RH]])
    
    # 2. Load the scaler and transform the data
    try:
        scaler = joblib.load("scaler.joblib")
        scaled_data = scaler.transform(data)
        
        # 3. Get prediction
        result = predict_power(scaled_data)
        st.success(f"Estimated Energy Output: {result}")
    except Exception as e:
        st.error(f"Make sure you uploaded 'scaler.joblib' to GitHub! Error: {e}")