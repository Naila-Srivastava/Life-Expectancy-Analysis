import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# Load the model and scaler
model = tf.keras.models.load_model("models/life_expectancy_model.keras")
scaler = joblib.load("models/scaler.pkl")

st.set_page_config(page_title="Life Expectancy Prediction", layout="centered")
st.title("🧬 Life Expectancy Predictor")

st.markdown("Predicts life expectancy based on health and socio-economic factors.")

# Sidebar inputs
st.sidebar.header("Input Features")

# Example feature list – make sure this order matches your training data
features = {
    "Adult Mortality": st.sidebar.slider("Adult Mortality", 0, 500, 150),
    "infant deaths": st.sidebar.slider("Infant Deaths", 0, 200, 10),
    "Alcohol": st.sidebar.slider("Alcohol Consumption", 0.0, 20.0, 5.0),
    "percentage expenditure": st.sidebar.slider("Percentage Expenditure", 0.0, 10000.0, 500.0),
    "Hepatitis B": st.sidebar.slider("Hepatitis B Vaccination (%)", 0, 100, 80),
    "Measles": st.sidebar.slider("Measles Cases", 0, 5000, 100),
    "BMI": st.sidebar.slider("BMI", 10.0, 50.0, 20.0),
    "under-five deaths": st.sidebar.slider("Under 5 Deaths", 0, 500, 50),
    "Polio": st.sidebar.slider("Polio (%)", 0, 100, 90),
    "Total expenditure": st.sidebar.slider("Total Health Expenditure (%)", 0.0, 20.0, 5.0),
    "Diphtheria": st.sidebar.slider("Diphtheria (%)", 0, 100, 85),
    "HIV/AIDS": st.sidebar.slider("HIV/AIDS Rate", 0.0, 5.0, 0.1),
    "GDP": st.sidebar.slider("GDP per capita", 0.0, 20000.0, 5000.0),
    "Population": st.sidebar.slider("Population (in millions)", 0.0, 1500000000.0, 100000000.0),
    "thinness 1-19 years": st.sidebar.slider("Thinness 1-19 years (%)", 0.0, 30.0, 5.0),
    "thinness 5-9 years": st.sidebar.slider("Thinness 5-9 years (%)", 0.0, 30.0, 5.0),
    "Income composition of resources": st.sidebar.slider("Income Composition", 0.0, 1.0, 0.5),
    "Schooling": st.sidebar.slider("Schooling (years)", 0.0, 20.0, 12.0),
    "Life expectancy": st.sidebar.slider("Previous Life Expectancy", 40.0, 90.0, 70.0)
}

# Create input dataframe
input_df = pd.DataFrame([features])

# Rename columns to match training format
input_df.columns = [
    'Year', 'Adult_Mortality', 'infant_deaths', 'Alcohol',
    'percentage_expenditure', 'Hepatitis_B', 'Measles', 'BMI',
    'under_five_deaths', 'Polio', 'Total_expenditure', 'Diphtheria',
    'HIV_AIDS', 'GDP', 'Population', 'thinness_1_19_years',
    'thinness_5_9_years', 'Income_composition_of_resources', 'Schooling'
]

# Now scale safely
scaled_input = scaler.transform(input_df)

# Debug print
st.write("**Scaled Input Features**:")
st.write(scaled_input)

# Debug prints
st.write("🔍 **Input Features Received**:")
st.write(input_df)

# Model prediction
raw_prediction = model.predict(scaled_input)

# Debug prediction shape/type
st.write("**Raw Model Prediction Output:**")
st.write(raw_prediction)
st.write(f"Type: {type(raw_prediction)}, Shape: {np.shape(raw_prediction)}")

# Flatten
try:
    prediction = float(raw_prediction[0]) if isinstance(raw_prediction[0], (list, np.ndarray)) else float(raw_prediction)
except Exception as e:
    st.error(f"Prediction parsing failed: {e}")
    prediction = "N/A"

# Final output
if isinstance(prediction, float):
    st.subheader("📉 Predicted Life Expectancy")
    st.success(f"Estimated Life Expectancy: **{prediction:.2f} years**")
else:
    st.error("⚠️ Prediction could not be computed. Please check inputs or model.")
