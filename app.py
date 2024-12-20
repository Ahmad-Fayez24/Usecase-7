import streamlit as st
import requests
import pandas as pd

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
    url = "https://usecase-7-k4mj.onrender.com/predict"
    # Prepare the payload for the API request
    payload = {"age": age, "appearance": appearance, "goals": goals}
    # Send the POST request to the FastAPI server
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        prediction = response.json().get("prediction")
        st.write(f"Prediction: {prediction}")
    else:
        st.write("Error:", response.status_code)
