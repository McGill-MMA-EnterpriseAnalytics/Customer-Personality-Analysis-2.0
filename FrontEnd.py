import streamlit as st
import requests
import json

# Define the URL of the MLflow model
MLFLOW_API_URL = "https://<databricks-instance>/api/2.0/mlflow/invocations"

# Streamlit app layout
st.title('ML Model Prediction Interface')

# Add tabs for different functionalities
tabs = ["Causal Inference", "Clustering", "Segment Prediction"]
selected_tab = st.sidebar.radio("Select Functionality", tabs)

# Display content based on selected tab
if selected_tab == "Causal Inference":
    st.write("Causal Inference tab content goes here")
elif selected_tab == "Clustering":
    st.write("Clustering tab content goes here")
elif selected_tab == "Segment Prediction":
    # Button to make prediction
    if st.button('Predict'):
        # Prepare the data in the format the MLflow model expects
        feature_1 = st.number_input("Enter feature 1")
        feature_2 = st.number_input("Enter feature 2")
        feature_3 = st.number_input("Enter feature 3")
        
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
