# Telecom Customer Churn Prediction Project

## Overview

This project focuses on predicting customer churn for Expresso Telecommunications Company using machine learning. The solution includes data cleaning, exploratory data analysis, model training, and deployment via a Streamlit web application.

## Project Structure

```
telecom-churn-prediction/
├── data/
│   ├── Expresso_churn_dataset.csv         # Raw dataset
│   └── expresso_cleaned.csv              # Cleaned dataset
├── models/
│   ├── logistic_regression_model.pkl     # Trained model
│   ├── scaler.pkl                       # Feature scaler
│   └── label_encoder.pkl                # Label encoders
├── notebooks/
│   └── Cleaning_The_DataSet.ipynb       # Data cleaning notebook
├── app.py                               # Streamlit application
├── requirements.txt                     # Python dependencies
└── README.md
```

## Key Steps

### 1. Data Cleaning

The raw dataset contained:
- 802,038 customer records with 19 features
- Significant missing values (especially in REGION column)
- Inconsistent data types
- Duplicate records

Cleaning steps included:
- Dropping irrelevant columns (user_id, ORANGE, TIGO, etc.)
- Handling missing values:
  - Filled missing REGION values with "NO-REGION"
  - Filled other numerical NaNs with 0
- Removing duplicate records (249,335 duplicates found)
- Final cleaned dataset: 552,703 records with 12 features

### 2. Exploratory Data Analysis

Key findings:
- Churn rate: ~18.7% (150,283 churned customers)
- Customers without region information had higher churn rates
- Dakar region had the most customers but not necessarily the highest churn rate
- Regularity (1-62) showed interesting patterns with churn

### 3. Feature Engineering

Selected features for modeling:
- REGION (categorical)
- TENURE (categorical)
- MONTANT (total amount)
- FREQUENCE_RECH (recharge frequency)
- REVENUE (monthly income)
- ARPU_SEGMENT (average revenue per user)
- FREQUENCE (call frequency)
- DATA_VOLUME (data usage)
- ON_NET (on-net calls)
- REGULARITY (1-62)
- FREQ_TOP_PACK (frequency of top package usage)

### 4. Model Building

Used Logistic Regression due to:
- Good performance on binary classification
- Interpretability of results
- Efficiency with the dataset size

Preprocessing:
- Label encoding for categorical variables (REGION, TENURE)
- Feature scaling for numerical variables

### 5. Streamlit Deployment

The web application allows users to:
- Input customer information through an intuitive form
- Get instant churn predictions
- View probability scores for better decision making

Features include:
- Dropdowns for categorical variables
- Number inputs for continuous features
- Clear prediction output with probabilities
- Informative sidebar with model details

## How to Use

### Local Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/telecom-churn-prediction.git
   cd telecom-churn-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

### Streamlit Cloud Deployment

1. Push the repository to GitHub
2. Connect to Streamlit Cloud
3. Deploy with the following files:
   - app.py
   - requirements.txt
   - All .pkl model files

## Key Files

- `Cleaning_The_DataSet.ipynb`: Jupyter notebook with complete data cleaning process
- `app.py`: Streamlit application code
- `expresso_cleaned.csv`: Cleaned dataset
- Model files (`*.pkl`): Serialized model and preprocessing objects

## Future Improvements

- Experiment with other algorithms (Random Forest, XGBoost)
- Add more visualizations to the Streamlit app
- Implement feature importance analysis
- Create customer segmentation based on churn risk

## Acknowledgments

Project created for Expresso Telecommunications Company to help reduce customer churn through data-driven insights.
