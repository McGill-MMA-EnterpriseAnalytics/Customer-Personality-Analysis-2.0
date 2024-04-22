
import streamlit as st
import pandas as pd

def display_data_catalog():
    # Read the data catalog file
    data_catalog = pd.read_csv('https://raw.githubusercontent.com/McGill-MMA-EnterpriseAnalytics/Customer-Personality-Analysis-2.0/main/Data/Preprocessed%20Data/Final%20Preprocessed%20Data.csv?token=GHSAT0AAAAAACL3VCJB4QQSO53AD2M7BBPIZRGSKYA')

    # Display the data catalog in a table
    st.table(data_catalog)

display_data_catalog()