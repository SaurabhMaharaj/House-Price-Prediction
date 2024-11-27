import streamlit as st
import joblib
import numpy as np


model = joblib.load('house_price_model1.pkl')


st.title("House Price Prediction App")


sqft_living = st.number_input("Square Footage (Living Area)", min_value=0, step=1)
bedrooms = st.number_input("Number of Bedrooms", min_value=0, step=1)
bathrooms = st.number_input("Number of Bathrooms", min_value=0.0, step=0.5)


if st.button("Predict"):

    features = np.array([sqft_living, bedrooms, bathrooms]).reshape(1, -1)
    

    prediction = model.predict(features)[0]
    

    st.success(f"The predicted house price is: ${prediction:,.2f}")
