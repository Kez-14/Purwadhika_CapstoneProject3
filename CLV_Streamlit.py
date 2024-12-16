import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import LabelEncoder
import joblib
import plotly.express as px

st.title('Customer Lifetime Value Prediction: Car Insurance')

df = pd.read_csv('data_customer_lifetime_value.csv')
st.write('Customer Demographics')

demographic_col = [
    'Vehicle Class', 'Coverage', 'Renew Offer Type', 
    'EmploymentStatus', 'Marital Status', 'Education'
]

fig, axes = plt.subplots(2, 3, figsize=(20, 8))
for i, column in enumerate(demographic_col):
    row = i // 3
    col = i % 3
    sns.countplot(x=column, data=df, ax=axes[row, col], palette='Set1')
    axes[row, col].set_title(f"Distribution of {column}")
    axes[row, col].set_xlabel(column)
    axes[row, col].set_ylabel('Total')
plt.tight_layout()
st.pyplot(fig)

def preprocess_input(data, encoders):
    for col, le in encoders.items():
        data[col] = le.transform(data[col])
    return data

# RandomForest model
model_rf_after_tuning = joblib.load('CLV_RFModel.pkl')  # Pastikan path yang benar

# Sidebar
st.sidebar.header("Please input customer's features")

vehicle_class     = st.sidebar.selectbox('Vehicle Class', ['Four-Door Car', 'Two-Door Car', 'SUV', 'Sports Car', 'Luxury SUV', 'Luxury Car'])
coverage          = st.sidebar.selectbox('Coverage', ['Extended', 'Basic', 'Premium'])
renew_offer       = st.sidebar.selectbox('Renew Offer Type', ['Offer1', 'Offer3', 'Offer2', 'Offer4'])
employment_status = st.sidebar.selectbox('Employment Status', ['Retired', 'Employed', 'Disabled', 'Medical Leave', 'Unemployed'])
marital_status    = st.sidebar.selectbox('Marital Status', ['Divorced', 'Married', 'Single'])
education         = st.sidebar.selectbox('Education', ['High School or Below', 'College', 'Master', 'Bachelor', 'Doctor'])
num_policies      = st.sidebar.number_input('Number of Policies', min_value=1, max_value=9, value=1)
monthly_premium   = st.sidebar.number_input('Monthly Premium Auto', min_value=0.0, step=1.0)
total_claim       = st.sidebar.number_input('Total Claim Amount', min_value=0.0, step=1.0)
income            = st.sidebar.number_input('Income', min_value=0.0, step=100.0)

# Data based on user input
data = pd.DataFrame({
    'Vehicle Class'       : [vehicle_class],
    'Coverage'            : [coverage],
    'Renew Offer Type'    : [renew_offer],
    'EmploymentStatus'    : [employment_status],
    'Marital Status'      : [marital_status],
    'Education'           : [education],
    'Number of Policies'  : [num_policies],
    'Monthly Premium Auto': [monthly_premium],
    'Total Claim Amount'  : [total_claim],
    'Income'              : [income],
})

# Prediction using train model
prediction = model_rf_after_tuning.predict(data)
st.write(f"Predicted Customer Lifetime Value: {prediction[0]:.2f}")

# Prediction visualization
fig, ax = plt.subplots(figsize=(7, 3))
sns.barplot(x=["Predicted CLV"], y=[prediction[0]], ax=ax, palette='viridis')
ax.set_title('Predicted Customer Lifetime Value')
ax.set_ylabel('Customer Lifetime Value (CLV)')
ax.set_ylim(0, max(prediction[0] * 1.2, 50000))  # Mengatur batas atas agar grafik terlihat jelas
st.pyplot(fig)