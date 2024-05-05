import streamlit as st
import requests
import json

# Define the URL of the MLflow model
prediction_api_URL = "http://localhost:8000/predict/"
Causal_api_URL = "http://localhost:8000/causal/"
cluster_api_URL = "http://localhost:8000/clsuter/"

# Streamlit app layout
st.title('Customer-Personality-Analysis-2.0 Customer Insight Hub')

# Add tabs for different functionalities
tabs = ["Causal Inference", "Clustering", "Segment Prediction"]
selected_tab = st.sidebar.radio("Select Model", tabs)

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
    # Layout adjustment for two columns
    col1, col2 = st.columns(2)
    
    # First column
    with col1:
        Total_amount = st.number_input("Total Amount", value=0.0)
        Is_Parent = st.selectbox("Is Parent", [True, False])
        Total_Children = st.number_input("Total Children", value=0, step=1)
        NumDealsPurchases = st.number_input("Number of Deals/Purchases", value=0, step=1)
    
    # Second column
    with col2:
        Income = st.number_input("Income", value=0.0)
        Family_Size = st.number_input("Family Size", value=0, step=1)
        NumWebVisitsMonth = st.number_input("Number of Web Visits per Month", value=0, step=1)
        Total_purchase = st.number_input("Total Purchase", value=0.0)
        MntWines = st.number_input("Amount Spent on Wines", value=0.0)
        Teenhome = st.number_input("Number of Teenagers at Home", value=0, step=1)

    # Button to make prediction
    if st.button('Predict'):
        # Prepare data in JSON format
        data = json.dumps({
            "Total_amount": Total_amount,
            "Is_Parent": Is_Parent,
            "Total_Children": Total_Children,
            "NumDealsPurchases": NumDealsPurchases,
            "Income": Income,
            "Family_Size": Family_Size,
            "NumWebVisitsMonth": NumWebVisitsMonth,
            "Total_purchase": Total_purchase,
            "MntWines": MntWines,
            "Teenhome": Teenhome
        })
        headers = {'Content-Type': 'application/json'}
        
        # Send the data to the model
        response = requests.post(prediction_api_URL, data=data, headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            st.success(f'Result: {result}')
        else:
            st.error('Failed to get prediction from the model.')
