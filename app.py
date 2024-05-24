import streamlit as st
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the model
with open('car_price_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the feature names
with open('feature_names.pkl', 'rb') as file:
    feature_names = pickle.load(file)

# Function to preprocess input data
def preprocess_input(data):
    input_df = pd.DataFrame([data])
    input_df = pd.get_dummies(input_df, columns=['brand', 'model', 'fuel_type'], drop_first=True)

    # Ensure all expected columns are present
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Reorder the columns to match the training order
    input_df = input_df[feature_names]
    
    return input_df

# Function to predict car price
def predict_price(model, input_data):
    preprocessed_data = preprocess_input(input_data)
    return model.predict(preprocessed_data)

# Streamlit application
st.title("Car Price Prediction")

# Collect user input
age = st.number_input('Car Age (years)', min_value=0, max_value=30, value=5)
mileage = st.number_input('Mileage (km)', min_value=0, max_value=300000, value=50000)
brand = st.selectbox('Brand', ['Toyota', 'Ford', 'BMW'])
model_name = st.selectbox('Model', ['Corolla', 'Camry', 'Focus', 'Mustang', 'X3'])
fuel_type = st.selectbox('Fuel Type', ['Petrol', 'Diesel', 'Electric'])

# Create a dictionary from the user input
input_data = {
    'age': age,
    'mileage': mileage,
    'brand': brand,
    'model': model_name,
    'fuel_type': fuel_type
}

# When the user clicks the button, predict the price
if st.button('Predict Price'):
    prediction = predict_price(model, input_data)
    st.write(f"The predicted price of the car is ${prediction[0]:.2f}")
