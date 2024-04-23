import streamlit as st
import requests
import json

# Define the URL of the MLflow model
MLFLOW_API_URL = "https://<databricks-instance>/api/2.0/mlflow/invocations"

# Streamlit app layout
st.title('ML Model Prediction Interface')
st.write('Please input the features for model prediction.')

# Collecting user inputs
feature_1 = st.number_input('Enter feature 1')
feature_2 = st.number_input('Enter feature 2')
feature_3 = st.number_input('Enter feature 3')

# Button to make prediction
if st.button('Predict'):
    # Prepare the data in the format the MLflow model expects
    data = json.dumps({
        "columns": ["feature_1", "feature_2", "feature_3"],
        "data": [[feature_1, feature_2, feature_3]]
    })
    headers = {'Content-Type': 'application/json'}
    
    # Send the data to the model
    response = requests.post(MLFLOW_API_URL, data=data, headers=headers)
    
    if response.status_code == 200:
        result = response.json()
        st.success(f'Result: {result}')
    else:
        st.error('Failed to get prediction from the model.')

# Displaying the model output
st.write('The model output will be displayed here once you input the features and click predict.')
