
import streamlit as st
import joblib
import pandas as pd

# Load trained ML model
model = joblib.load("water_model.pkl")

# Page configuration
st.set_page_config(
    page_title="Water Potability Predictor",
    page_icon="💧",
    layout="centered"
)

# App heading
st.title("💧 Water Potability Predictor")

st.write(
    "Enter the water quality parameters below "
    "to predict whether the water is potable or not."
)

st.divider()

# Input section
st.subheader("Water Quality Parameters")

col1, col2 = st.columns(2)

with col1:
    ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0)
    hardness = st.number_input("Hardness", min_value=0.0, value=200.0)
    solids = st.number_input("Solids", min_value=0.0, value=20000.0)
    chloramines = st.number_input("Chloramines", min_value=0.0, value=7.0)
    sulfate = st.number_input("Sulfate", min_value=0.0, value=300.0)

with col2:
    conductivity = st.number_input("Conductivity", min_value=0.0, value=400.0)
    organic_carbon = st.number_input("Organic Carbon", min_value=0.0, value=10.0)
    trihalomethanes = st.number_input("Trihalomethanes", min_value=0.0, value=60.0)
    turbidity = st.number_input("Turbidity", min_value=0.0, value=4.0)

st.divider()

# Prediction button
if st.button("Predict Water Quality", type="primary", use_container_width=True):

    # Prepare input data
    input_data = pd.DataFrame([{
        "ph": ph,
        "Hardness": hardness,
        "Solids": solids,
        "Chloramines": chloramines,
        "Sulfate": sulfate,
        "Conductivity": conductivity,
        "Organic_carbon": organic_carbon,
        "Trihalomethanes": trihalomethanes,
        "Turbidity": turbidity
    }])

    # Make prediction
    prediction = model.predict(input_data)

    # Display result
    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.success("Water is Potable 💧")
    else:
        st.error("Water is Not Potable ⚠️")

    st.caption(
        "This is a machine learning prediction, "
        "not a laboratory water safety test."
    )
