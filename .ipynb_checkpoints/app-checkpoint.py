import streamlit as st
from prediction import predict_power


st.title("Power Plant Prediction")

AT = st.number_input("Temperature")
V = st.number_input("Vacuum")
AP = st.number_input("Pressure")
RH = st.number_input("Humidity")


if st.button("Predict"):

    data = [[AT, V, AP, RH]]

    result = predict_power(data)

    st.success(result)