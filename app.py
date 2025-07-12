# Import the necessary packages

import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("models/life_expectancy_model.keras")
scaler = joblib.load("models/scaler.pkl")

st.set_page_config(page_title="Life Expectancy Predictor", layout="wide")

st.title("🌍 Life Expectancy Prediction App")
st.markdown("Provide country health & economic data below to predict life expectancy.")

# Sidebar for user input
st.sidebar.header("Input Features")

def user_input_features():
    year = st.sidebar.slider("Year", 2000, 2020, 2015)
    adult_mortality = st.sidebar.slider("Adult Mortality", 0, 1000, 150)
    infant_deaths = st.sidebar.slider("Infant Deaths", 0, 300, 10)
    alcohol = st.sidebar.slider("Alcohol", 0.0, 20.0, 5.0)
    percentage_expenditure = st.sidebar.slider("Health Expenditure %", 0.0, 100000.0, 5000.0)
    hepatitis_b = st.sidebar.slider("Hepatitis B", 0, 100, 80)
    measles = st.sidebar.slider("Measles Cases", 0, 10000, 50)
    bmi = st.sidebar.slider("BMI", 10.0, 50.0, 22.0)
    under_five_deaths = st.sidebar.slider("Under-5 Deaths", 0, 400, 20)
    polio = st.sidebar.slider("Polio Immunization %", 0, 100, 90)
    total_expenditure = st.sidebar.slider("Total Health Expenditure", 0.0, 20.0, 6.0)
    diphtheria = st.sidebar.slider("Diphtheria %", 0, 100, 85)
    hiv_aids = st.sidebar.slider("HIV/AIDS", 0.0, 100.0, 1.0)
    gdp = st.sidebar.slider("GDP", 0.0, 100000.0, 3000.0)
    population = st.sidebar.slider("Population", 0.0, 1500000000.0, 5000000.0)
    thinness_1_19 = st.sidebar.slider("Thinness 10-19 yrs", 0.0, 25.0, 5.0)
    thinness_5_9 = st.sidebar.slider("Thinness 5-9 yrs", 0.0, 25.0, 5.0)
    income_comp = st.sidebar.slider("Income Composition of Resources", 0.0, 1.0, 0.6)
    schooling = st.sidebar.slider("Schooling Years", 0.0, 20.0, 12.0)

    data = {
        'Year': year,
        'Adult_Mortality': adult_mortality,
        'infant_deaths': infant_deaths,
        'Alcohol': alcohol,
        'percentage_expenditure': percentage_expenditure,
        'Hepatitis_B': hepatitis_b,
        'Measles': measles,
        'BMI': bmi,
        'under-five_deaths': under_five_deaths,
        'Polio': polio,
        'Total_expenditure': total_expenditure,
        'Diphtheria': diphtheria,
        'HIV/AIDS': hiv_aids,
        'GDP': gdp,
        'Population': population,
        'thinness__1-19_years': thinness_1_19,
        'thinness_5-9_years': thinness_5_9,
        'Income_composition_of_resources': income_comp,
        'Schooling': schooling
    }

    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Preprocess
scaled_input = scaler.transform(input_df)

# Predict
prediction = model.predict(scaled_input)[0]

st.subheader("📉 Predicted Life Expectancy")
st.success(f"Estimated Life Expectancy: **{prediction:.2f} years**")