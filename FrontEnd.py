import streamlit as st
import requests
import json

# Define the URL of the MLflow model
prediction_api_URL = "https://<databricks-instance>/api/2.0/mlflow/invocations"
Causal_api_URL = "https://<databricks-instance>/api/2.0/mlflow/invocations"
cluster_api_URL = "https://<databricks-instance>/api/2.0/mlflow/invocations"

# Streamlit app layout
st.title('Customer-Personality-Analysis-2.0 Customer Insight Hub')

# Add tabs for different functionalities
tabs = ["Causal Inference", "Clustering", "Segment Prediction"]
selected_tab = st.sidebar.radio("Select Functionality", tabs)

# Display content based on selected tab
if selected_tab == "Causal Inference":
    # Prepare the data in the format the MLflow model expects
    target_options = ['Recency', 'Total_purchase', 'Total_amount']
    Target = st.selectbox("Select Target", target_options)
    
    treatment_options = ['Income_Category_High', 'Is_Parent', 'Cmp_Attitude', 'Complain']
    Treatment = st.selectbox("Select Treatment", treatment_options)
    
    confounding_options = ['Income_Category_High', 'Income_Category_Low', 'Income_Category_Medium',
                           'Complain', 'Is_Parent', 'Cmp_Attitude', 'Family_Size', 'Age',
                           'Member_Year', 'Total_amount', 'Total_purchase',
                           'NumWebVisitsMonth', 'NumDealsPurchases', 'Recency']
    ConfoundingVar = st.multiselect("Select Confounding Variables", confounding_options)
    
    # Button to call the model
    if st.button('Run Causal Inference Model'):
        data = json.dumps({
            "columns": ["Target", "Treatment", "Confounding Variables"],
            "data": [[Target, Treatment, ConfoundingVar]]
        })
        headers = {'Content-Type': 'application/json'}
        
        # Send the data to the model
        response = requests.post(Causal_api_URL, data=data, headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            st.success(f'Result: {result}')
        else:
            st.error('Failed to get prediction from the model.')
elif selected_tab == "Clustering":
    # Button to run clustering
    if st.button('Clustering'):
        # Send the data to the model
        response = requests.post(cluster_api_URL)
        
        if response.status_code == 200:
            result = response.json()
            st.success(f'Result: {result}')
        else:
            st.error('Failed to get prediction from the model.')
elif selected_tab == "Segment Prediction":
    # Button to make prediction
    if st.button('Predict'):
        # Send the data to the model
        response = requests.post(prediction_api_URL)
        
        if response.status_code == 200:
            result = response.json()
            st.success(f'Result: {result}')
        else:
            st.error('Failed to get prediction from the model.')
