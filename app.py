import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

# Load model and scaler
model = tf.keras.models.load_model("models/life_expectancy_model.keras")
scaler = joblib.load("models/scaler.pkl")

# Streamlit UI
st.set_page_config(page_title="Life Expectancy Predictor", layout="centered")
st.title("🌍 Life Expectancy Predictor")
st.write("Predict a country's average life expectancy based on various health and economic indicators.")

# Sidebar inputs
st.sidebar.header("Input Features")

year = st.sidebar.slider("Year", 2000, 2015, 2010)
adult_mortality = st.sidebar.slider("Adult Mortality", 0, 500, 150)
infant_deaths = st.sidebar.slider("Infant Deaths", 0, 100, 5)
alcohol = st.sidebar.slider("Alcohol Consumption", 0.0, 20.0, 5.0)
percentage_expenditure = st.sidebar.slider("Percentage Expenditure", 0.0, 10000.0, 500.0)
hepatitis_b = st.sidebar.slider("Hepatitis B Immunization (%)", 0, 100, 80)
measles = st.sidebar.slider("Measles Cases", 0, 10000, 50)
bmi = st.sidebar.slider("Average BMI", 10.0, 50.0, 25.0)

# 🚨 Warning if BMI<10
if bmi < 10:
    st.warning("⚠️ BMI below 10 is not realistic. Minimum allowed is 10 to avoid app errors.")
    
under_five_deaths = st.sidebar.slider("Under-Five Deaths", 0, 100, 10)
polio = st.sidebar.slider("Polio Immunization (%)", 0, 100, 90)
total_expenditure = st.sidebar.slider("Total Expenditure (%)", 0.0, 20.0, 5.0)
diphtheria = st.sidebar.slider("Diphtheria Immunization (%)", 0, 100, 85)
hiv_aids = st.sidebar.slider("HIV/AIDS Prevalence", 0.0, 50.0, 1.0)
gdp = st.sidebar.slider("GDP (log scale)", 0.0, 200000.0, 5000.0)
population = st.sidebar.slider("Population", 0.0, 1e9, 1e7)
thinness_1_19 = st.sidebar.slider("Thinness 10-19 yrs", 0.0, 25.0, 5.0)
thinness_5_9 = st.sidebar.slider("Thinness 5-9 yrs", 0.0, 25.0, 5.0)
income_comp = st.sidebar.slider("Income Composition of Resources", 0.0, 1.0, 0.7)
schooling = st.sidebar.slider("Schooling Years", 0.0, 20.0, 12.0)

# Combine inputs into DataFrame
input_df = pd.DataFrame([[
    year, adult_mortality, infant_deaths, alcohol, percentage_expenditure,
    hepatitis_b, measles, bmi, under_five_deaths, polio, total_expenditure,
    diphtheria, hiv_aids, gdp, population, thinness_1_19, thinness_5_9,
    income_comp, schooling
]])

# Set correct column names matching the scaler's training
input_df.columns = [
    'Year', 'Adult_Mortality', 'infant_deaths', 'Alcohol',
    'percentage_expenditure', 'Hepatitis_B', 'Measles', 'BMI',
    'under-five_deaths', 'Polio', 'Total_expenditure', 'Diphtheria',
    'HIV/AIDS', 'GDP', 'Population', 'thinness__1-19_years',
    'thinness_5-9_years', 'Income_composition_of_resources', 'Schooling'
]

# Scale input
scaled_input = scaler.transform(input_df)

# Predict
prediction = model.predict(scaled_input)[0][0]

# Show result
st.subheader("📉 Predicted Life Expectancy")
st.success(f"Estimated Life Expectancy: **{prediction:.2f} years**")
