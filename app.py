import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression

# Set page configuration
st.set_page_config(page_title="Telecom Churn Prediction", page_icon="📡", layout="wide")

# Load model and preprocessors
try:
    with open('logistic_regression_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    with open('label_encoder.pkl', 'rb') as file:
        encoder_dict = pickle.load(file)
except FileNotFoundError:
    st.error("Model, scaler, or encoder files not found. Ensure all .pkl files are in the directory.")
    st.stop()

# Custom CSS for better visuals
st.markdown("""
<style>
    .main {background-color: #f5f5f5;}
    .stButton>button {background-color: #0066cc; color: white; border-radius: 8px; padding: 10px 20px;}
    .stButton>button:hover {background-color: #0055aa;}
    .stNumberInput input {border-radius: 5px;}
    .stSelectbox {border-radius: 5px;}
    .sidebar .sidebar-content {background-color: #e6f3ff;}
</style>
""", unsafe_allow_html=True)

# Title and header
st.title("📡 Telecom Customer Churn Prediction")
st.markdown("Enter customer details to predict the likelihood of churn.")

# Input form
with st.form("churn_form"):
    st.subheader("Customer Information")
    col1, col2 = st.columns(2)

    with col1:
        region_options = list(encoder_dict['REGION'].classes_)
        REGION = st.selectbox('Region', region_options, help="Select customer's region")
        region_encoded = encoder_dict['REGION'].transform([REGION])[0]

        tenure_options = list(encoder_dict['TENURE'].classes_)
        TENURE = st.selectbox('Tenure', tenure_options, help="Select customer's tenure")
        tenure_encoded = encoder_dict['TENURE'].transform([TENURE])[0]

        montant = st.number_input('Montant (Amount)', min_value=0.0, step=0.1, format="%.2f", help="Total amount spent")
        frequence_rech = st.number_input('Recharge Frequency', min_value=0.0, step=0.1, format="%.2f", help="Frequency of recharges")
        revenue = st.number_input('Revenue', min_value=0.0, step=0.1, format="%.2f", help="Customer revenue")

    with col2:
        arpu_segment = st.number_input('ARPU Segment', min_value=0.0, step=0.1, format="%.2f", help="Average revenue per user segment")
        frequence = st.number_input('Frequency', min_value=0.0, step=0.1, format="%.2f", help="Usage frequency")
        data_volume = st.number_input('Data Volume', min_value=0.0, step=0.1, format="%.2f", help="Data usage volume")
        on_net = st.number_input('On-Net Calls', min_value=0.0, step=0.1, format="%.2f", help="On-network call volume")
        regularity = st.number_input('Regularity', min_value=1.0, max_value=62.0, step=1.0, format="%.1f", help="Usage regularity (1-62)")
        freq_top_pack = st.number_input('Top Pack Frequency', min_value=0.0, step=0.1, format="%.2f", help="Frequency of top pack usage")

    # Predict button
    submitted = st.form_submit_button("Predict Churn", use_container_width=True)

# Prediction logic
if submitted:
    features = np.array([region_encoded, tenure_encoded, montant, frequence_rech, revenue, arpu_segment,
                         frequence, data_volume, on_net, regularity, freq_top_pack]).reshape(1, -1)
    scaled_features = scaler.transform(features)

    try:
        prediction = model.predict(scaled_features)[0]
        proba = model.predict_proba(scaled_features)[0][1]

        st.subheader("Prediction Result")
        if prediction == 1:
            st.error("Customer is likely to churn.")
        else:
            st.success("Customer is likely to stay.")
        st.metric("Churn Probability", f"{proba:.2%}")
    except Exception as e:
        st.error(f"Prediction error: {e}")

# Sidebar with model information
with st.sidebar:
    st.header("About This Model")
    st.markdown("""
    This model predicts customer churn for Espresso Telecommunications using a Logistic Regression algorithm. It analyzes:
    - Region
    - Tenure
    - Montant
    - Recharge Frequency
    - Revenue
    - ARPU Segment
    - Usage Frequency
    - Data Volume
    - On-Net Calls
    - Regularity
    - Top Pack Frequency
    """)
    st.markdown("**Developed for Espresso Telecommunications**")

# Footer
st.markdown("---")
st.markdown("© 2025 Espresso Telecommunications Company")
