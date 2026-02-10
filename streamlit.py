import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("life_expectancy_model.pkl")
scaler = joblib.load("scaler (2).pkl")

st.set_page_config(page_title="Life Expectancy Predictor", layout="wide")

st.title("🌍 Life Expectancy Prediction")
st.write("Fill all country details to predict life expectancy")

st.subheader("📊 Country Information")

col1, col2, col3 = st.columns(3)

with col1:
    year = st.slider("Year", 2000, 2030, 2015)
    status = st.selectbox("Country Status", ["Developing", "Developed"])
    adult_mortality = st.slider("Adult Mortality", 1, 750, 150)
    infant_deaths = st.slider("Infant Deaths", 0, 1000, 5)
    alcohol = st.slider("Alcohol Consumption", 0.0, 20.0, 4.5)

with col2:
    expenditure = st.slider("Health Expenditure", 0.0, 20000.0, 500.0)
    hepatitis = st.slider("Hepatitis B Immunization (%)", 0, 100, 85)
    measles = st.slider("Measles Cases", 0, 200000, 50)
    bmi = st.slider("Average BMI", 10.0, 50.0, 25.0)
    under_five = st.slider("Under-five Deaths", 0, 1000, 6)

with col3:
    polio = st.slider("Polio Immunization (%)", 0, 100, 90)
    total_exp = st.slider("Total Health Expenditure (%)", 0.0, 20.0, 6.0)
    diphtheria = st.slider("Diphtheria Immunization (%)", 0, 100, 90)
    hiv = st.slider("HIV/AIDS Rate", 0.0, 50.0, 0.1)
    gdp = st.slider("GDP", 0.0, 120000.0, 8000.0)

st.subheader("📈 Socio-Economic Factors")

col4, col5 = st.columns(2)

with col4:
    population = st.number_input("Population", 1000, 2000000000, 5000000)
    thin1 = st.slider("Thinness 1-19 years (%)", 0.0, 30.0, 2.5)
    thin2 = st.slider("Thinness 5-9 years (%)", 0.0, 30.0, 2.3)

with col5:
    income = st.slider("Income Composition of Resources", 0.0, 1.0, 0.75)
    schooling = st.slider("Schooling Years", 0.0, 25.0, 14.0)

# Encode status
status = 1 if status == "Developed" else 0

# -------- Prediction Button --------
st.write("")
if st.button("🔮 Predict Life Expectancy"):
    input_data = [[
        year, status, adult_mortality, infant_deaths, alcohol, expenditure,
        hepatitis, measles, bmi, under_five, polio, total_exp,
        diphtheria, hiv, gdp, population, thin1, thin2, income, schooling
    ]]

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    st.success(f"🎉 Predicted Life Expectancy: {prediction[0]:.2f} years")
