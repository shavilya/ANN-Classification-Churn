import streamlit as st 
import pandas as pd 
import numpy as np 

import tensorflow as tf 
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder 
import pickle

# Load the trained model 
model = tf.keras.models.load_model('model.h5')

# Load the encoders and scaler 
with open('label_encoder_gender.pkl', 'rb') as file: 
    label_encoder_gender = pickle.load(file)
    
with open('ohe_geo.pkl', 'rb') as file: 
    ohe_geo = pickle.load(file)
    
with open('scaler.pkl', 'rb') as file: 
    scaler = pickle.load(file)
    
# Streamlit Application 
st.title("Customer Churn Prediction")

# User Inputs 
geography = st.selectbox('Geography', ohe_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 100)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card?', [0, 1])
is_active_member = st.selectbox('Is Active Member?', [0, 1])

# Create input DataFrame 
input_data_df = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# OHE encode 'Geography'
geo_encoded = ohe_geo.transform(input_data_df[['Geography']]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=ohe_geo.get_feature_names_out(['Geography']))

# Label encode 'Gender'
input_data_df['Gender'] = label_encoder_gender.transform(input_data_df['Gender'])

# Drop original 'Geography' column and combine encoded geography
input_data_df = input_data_df.drop(columns=['Geography'])
input_data_df = pd.concat([input_data_df.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data (only on numerical + encoded columns)
input_data_scaled = scaler.transform(input_data_df)

# Prediction Churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f"Churn Probability: {prediction_proba:.2f}")
# Display result
if prediction_proba > 0.5:
    st.write("The Customer is likely to churn")
else:
    st.write("The Customer is not likely to churn")

