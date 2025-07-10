import streamlit as st
import numpy as np
import joblib

# Load the saved model
model = joblib.load('house_price_model.pkl')

st.title("House Price Predictor")

MedInc = st.slider('Median Income', 0.5, 15.0, 3.5)
HouseAge = st.slider('House Age', 1, 50, 20)
AveRooms = st.slider('Average Rooms', 1, 15, 5)
AveBedrms = st.slider('Average Bedrooms', 1, 5, 2)
Population = st.slider('Population', 100, 5000, 1500)
AveOccup = st.slider('Average Occupants', 0.5, 10.0, 3.0)
Latitude = st.slider('Latitude', 32.0, 42.0, 37.5)
Longitude = st.slider('Longitude', -124.0, -114.0, -118.0)

if st.button('Predict Price'):
    features = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])
    price = model.predict(features)[0]
    st.success(f"Predicted House Price: ${price * 100000:.2f}")