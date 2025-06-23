import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression

# Load the trained model, scaler, and encoder
try:
    with open('logistic_regression_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    with open('label_encoder.pkl', 'rb') as file:
        encoder_dict = pickle.load(file)
except FileNotFoundError:
    st.error("Model, scaler, or encoder files not found. Ensure all .pkl files exist.")
    st.stop()

st.title('Telecom Customer Churn Prediction')
st.write('Enter customer information to predict churn:')

# Input fields
region_options = list(encoder_dict['REGION'].classes_)
REGION = st.selectbox('Region', region_options)
region_encoded = encoder_dict['REGION'].transform([REGION])[0]

tenure_options = list(encoder_dict['TENURE'].classes_)
TENURE = st.selectbox('Tenure', tenure_options)
tenure_encoded = encoder_dict['TENURE'].transform([TENURE])[0]

montant = st.number_input('Montant', min_value=0.0)
frequence_rech = st.number_input('Frequence Rech', min_value=0.0)
revenue = st.number_input('Revenue', min_value=0.0)
arpu_segment = st.number_input('ARPU Segment', min_value=0.0)
frequence = st.number_input('Frequence', min_value=0.0)
data_volume = st.number_input('Data Volume', min_value=0.0)
on_net = st.number_input('On Net', min_value=0.0)
regularity = st.number_input('Regularity', min_value=1.0, max_value=62.0)
freq_top_pack = st.number_input('Freq Top Pack', min_value=0.0)

# Prediction button
if st.button('Predict Churn'):
    features = np.array([region_encoded, tenure_encoded, montant, frequence_rech, revenue, arpu_segment,
                         frequence, data_volume, on_net, regularity, freq_top_pack]).reshape(1, -1)
    try:
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)[0]
        proba = model.predict_proba(scaled_features)[0][1]
        st.write(f'Prediction: Customer is {"likely to churn" if prediction == 1 else "likely to stay"}.')
        st.write(f'Probability of Churning: {proba:.2%}')
    except Exception as e:
        st.error(f"Prediction error: {e}")

# Sidebar info
st.sidebar.header('About this model')
st.sidebar.write('''
This model predicts customer churn based on:
- Region
- Tenure
- Montant
- Frequence_rech
- Revenue
- ARPU_segment
- Frequence
- Data_volume
- On_net
- Regularity
- Freq_top_pack
Uses Logistic Regression trained on customer churn data.
''')

# Footer
st.markdown('Created for Expresso Telecommunications Company')
