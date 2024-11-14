import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained model
model = joblib.load('model.joblib')

st.title("Prediction Interface")

# Input fields for user data
age = st.number_input("Age", min_value=0, max_value=100, step=1)
appearance = st.number_input("Number of Appearances", min_value=0, step=1)
goals = st.number_input("Goals", min_value=0, step=1)

# Prepare input data for prediction
input_data = pd.DataFrame([[age, appearance, goals]], columns=["age", "appearance", "goals"])

# Display the input data to the user
st.write("Input Data:")
st.write(input_data)

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.write(f"Prediction: {prediction[0]}")
