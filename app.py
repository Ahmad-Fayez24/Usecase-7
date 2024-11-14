import streamlit as st
import pandas as pd
import pickle

# Load the pre-trained model
# Assuming you have saved the model as 'model.pkl' after training
# Load the model here; update the path if it's saved in another location
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Prediction Interface")

# Input fields for user data
age = st.number_input("Age", min_value=0, max_value=100, step=1)
appearance = st.selectbox("Appearance", ["Poor", "Average", "Good", "Excellent"])
goals = st.text_input("Goals")

# Map categorical features if needed (e.g., if 'appearance' needs encoding)
# appearance_dict = {"Poor": 0, "Average": 1, "Good": 2, "Excellent": 3}
# appearance_encoded = appearance_dict[appearance]

# Prepare input data for prediction
# Adjust feature order and names according to model requirements
input_data = pd.DataFrame([[age, appearance, goals]], columns=["age", "appearance", "goals"])

# Display the input data to the user
st.write("Input Data:")
st.write(input_data)

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.write(f"Prediction: {prediction[0]}")
