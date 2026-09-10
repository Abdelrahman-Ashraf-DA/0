import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Flight Delay Predictor", layout="centered")

# Load model artifacts
with open("flight_delay_artifacts.pkl", "rb") as f:
    artifacts = pickle.load(f)

lr = artifacts["lr_model"]
rf = artifacts["rf_model"]
sc = artifacts["scaler"]
model_columns = artifacts["model_columns"]
top_origin = artifacts["top_origin"]
top_dest = artifacts["top_dest"]

carriers = ['Alaska Airlines', 'Allegiant Air', 'American Airlines', 'Delta Air Lines',
            'Envoy Air', 'Frontier Airlines', 'JetBlue Airways', 'PSA Airlines',
            'Republic Airways', 'SkyWest Airlines', 'Southwest Airlines',
            'Spirit Airlines', 'United Airlines']

time_blocks = ['00:01-05:59','06:00-06:59','07:00-07:59','08:00-08:59','09:00-09:59',
               '10:00-10:59','11:00-11:59','12:00-12:59','13:00-13:59','14:00-14:59',
               '15:00-15:59','16:00-16:59','17:00-17:59','18:00-18:59','19:00-19:59',
               '20:00-20:59','21:00-21:59','22:00-22:59','23:00-23:59']

st.title("Flight Delay Predictor")
st.write("Predict whether a flight will be Delayed (Arrival Delay ≥ 15 Minutes)")

# Sidebar ==> model picker
model_choice = st.sidebar.selectbox("Choose Model", ["Random Forest", "Logistic Regression"])

st.subheader("Flight Details")

col1, col2 = st.columns(2)

with col1:
    carrier = st.selectbox("Operating Carrier", carriers)
    origin = st.selectbox("Origin Airport", top_origin + ["Other"])
    dest = st.selectbox("Destination Airport", top_dest + ["Other"])
    arr_block = st.selectbox("Arrival Time Block", time_blocks)
    dep_hour = st.slider("Scheduled Departure Hour", 0, 23, 9)

with col2:
    arr_hour = st.slider("Scheduled Arrival Hour", 0, 23, 11)
    taxi_out = st.number_input("Taxi Out (minutes)", min_value=0.0, max_value=200.0, value=15.0)
    distance = st.number_input("Distance (miles)", min_value=0, max_value=6000, value=500)
    elapsed_time = st.number_input("Actual Elapsed Time (minutes)", min_value=0, max_value=800, value=120)
    dep_delay_flag = st.selectbox("Departure Delay ≥ 15 Minutes?", ["No", "Yes"])
    cancelled_flag = st.selectbox("Cancelled?", ["No", "Yes"])

predict_btn = st.button("Predict")

if predict_btn:
    # Build one raw row same shape as training df before get_dummies
    row = {
        "Operating Carrier": carrier,
        "ORIGIN": origin,
        "DEST": dest,
        "Departure Delay ≥ 15 Minutes": 1 if dep_delay_flag == "Yes" else 0,
        "TAXI_OUT": taxi_out,
        "Arrival Time Block": arr_block,
        "CANCELLED": 1 if cancelled_flag == "Yes" else 0,
        "ACTUAL_ELAPSED_TIME": elapsed_time,
        "DISTANCE": distance,
        "DEP_HOUR": dep_hour,
        "ARR_HOUR": arr_hour
    }

    input_df = pd.DataFrame([row])
    input_encoded = pd.get_dummies(input_df, drop_first=True, dtype=int)

    # reindex ==> match training columns exactly, fill missing dummy cols with 0
    input_final = input_encoded.reindex(columns=model_columns, fill_value=0)

    if model_choice == "Logistic Regression":
        input_scaled = sc.transform(input_final)
        pred = lr.predict(input_scaled)[0]
        prob = lr.predict_proba(input_scaled)[:, 1][0]
    else:
        pred = rf.predict(input_final)[0]
        prob = rf.predict_proba(input_final)[:, 1][0]

    st.subheader("Result")
    if pred == 1:
        st.error(f"Prediction: DELAYED (probability of delay: {prob:.2%})")
    else:
        st.success(f"Prediction: ON TIME (probability of delay: {prob:.2%})")

    st.progress(float(prob))
