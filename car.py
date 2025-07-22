import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os
from PIL import Image
import locale

# ========== File Check ==========
assert os.path.exists("best_model.pkl"), "❌ best_model.pkl not found!"
assert os.path.exists("model_features.pkl"), "❌ model_features.pkl not found!"
assert os.path.exists("feature_importance.csv"), "❌ feature_importance.csv not found!"

# ========== Load Model and Assets ==========
model = joblib.load("best_model.pkl")
feature_columns = joblib.load("model_features.pkl")
feature_importance = pd.read_csv("feature_importance.csv")  # Should have columns: Feature, Importance

# ========== App Config ==========
st.set_page_config(page_title="Car Price Prediction", layout="wide")
st.title("🚗 Car Price Prediction App")

# ========== Sidebar Inputs ==========
st.sidebar.header("🔧 Input Vehicle Features")

make = st.sidebar.selectbox("Make", [
    'Maruti', 'Ford', 'Tata', 'Hyundai', 'Jeep', 'Datsun', 'Honda',
    'Mahindra', 'Mercedes-Benz', 'BMW', 'Renault', 'Audi', 'Toyota',
    'Mini', 'Kia', 'Skoda', 'Volkswagen', 'Volvo', 'MG', 'Nissan',
    'Fiat', 'Mahindra Ssangyong', 'Mitsubishi', 'Jaguar', 'Land Rover',
    'Chevrolet', 'Citroen', 'Opel', 'Mahindra Renault', 'Isuzu',
    'Lexus', 'Porsche', 'Hindustan Motors'
])

year = st.sidebar.slider("Year", 2000, 2025, 2018)
kms = st.sidebar.number_input("Kilometers Driven", 0, 500000, 10000)
fuel = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
transmission = st.sidebar.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.sidebar.selectbox("Owner Type", ['1st Owner', '2nd Owner', '3rd Owner', '4th Owner', '5th Owner'])
sunroof = st.sidebar.radio("Sun Roof", ["Yes", "No"])

# ========== Extra Features ==========
engine_cc = st.sidebar.number_input("Engine (CC)", min_value=500, max_value=5000, value=1500, step=100)
mileage_kmpl = st.sidebar.number_input("Mileage (kmpl)", min_value=0, max_value=50, value=15, step=1)
torque_nm = st.sidebar.number_input("Torque (Nm)", min_value=0, max_value=1000, value=150, step=10)
seats = st.sidebar.selectbox("Seats", [2, 4, 5, 6, 7, 8, 9, 10])
body_type = st.sidebar.selectbox("Car Body Type", [
    'Hatchback', 'SUV', 'Sedan', 'MUV', 'Coupe', 'Minivans',
    'Pickup Trucks', 'Convertibles', 'Hybrids', 'Wagon'
])
city = st.sidebar.selectbox("City", ['bangalore', 'chennai', 'delhi', 'hyderabad', 'jaipur', 'kolkata'])

# ========== Prediction ==========
if st.sidebar.button("🚀 Predict Price"):
    # Prepare input
    user_input = pd.DataFrame({
        'Make': [make],
        'Year': [year],
        'Kilometers_Driven': [kms],
        'Fuel Type': [fuel],
        'Transmission': [transmission],
        'Owner': [owner],
        'Sun Roof': [1 if sunroof == 'Yes' else 0],
        'Engine CC': [engine_cc],
        'Mileage KMPL': [mileage_kmpl],
        'Torque Nm': [torque_nm],
        'Seats': [seats],
        'Car Body Type': [body_type],
        'City': [city]
    })

    # One-hot encode
    user_encoded = pd.get_dummies(user_input)

    # Align with training features
    aligned_input = pd.DataFrame(0, index=[0], columns=feature_columns)
    for col in user_encoded.columns:
        if col in aligned_input.columns:
            aligned_input[col] = user_encoded[col]

    # Predict
    try:
        prediction = model.predict(aligned_input)[0]
        locale.setlocale(locale.LC_ALL, 'en_IN')
        price_str = locale.format_string("%d", int(prediction), grouping=True)
        st.success(f"💰 Estimated Car Price: ₹ {price_str}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ========== Feature Importance ==========
st.markdown("### 📊 Feature Importance (Model-Based)")
fig = px.bar(
    feature_importance.sort_values(by='Importance', ascending=True),
    x='Importance',
    y='Feature',
    orientation='h',
    color_discrete_sequence=['#009999']
)
fig.update_layout(height=500, template="plotly_white")
st.plotly_chart(fig, use_container_width=True)
