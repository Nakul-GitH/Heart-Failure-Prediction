import streamlit as st
import joblib
import numpy as np
import sys

st.title("Heart Failure Prediction")

# Collect user inputs for all features except DEATH_EVENT (target)
age = st.number_input('Age', min_value=18, max_value=90, value=20)
anaemia = st.selectbox('Anaemia', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
creatinine_phosphokinase = st.number_input('Creatinine Phosphokinase', min_value=0, value=100)
diabetes = st.selectbox('Diabetes', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
ejection_fraction = st.number_input('Ejection Fraction', min_value=0, max_value=100, value=50)
high_blood_pressure = st.selectbox('High Blood Pressure', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
platelets = st.number_input('Platelets', min_value=0, value=250000)
serum_creatinine = st.number_input('Serum Creatinine', min_value=0.0, value=1.0, step=0.01)
serum_sodium = st.number_input('Serum Sodium', min_value=100, max_value=200, value=135)
sex = st.selectbox('Sex', options=[0, 1], format_func=lambda x: 'Male' if x == 1 else 'Female')
smoking = st.selectbox('Smoking', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
time = st.number_input('Follow-up Period (days)', min_value=0, value=100)

# Example: To use these inputs for prediction, collect them in a list or array
user_input = np.array([[age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction,
                        high_blood_pressure, platelets, serum_creatinine, serum_sodium,
                        sex, smoking, time]])

# Add a prediction button
if st.button('Predict'):
    # Load the trained model
    model = joblib.load('heart_failure_model.pkl')
    # Make prediction
    prediction = model.predict(user_input)
    # Display the result
    if prediction[0] == 1:
        st.success('Prediction: High risk of death event')
    else:
        st.info('Prediction: Low risk of death event')

