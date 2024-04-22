
import streamlit as st
import pandas as pd

def display_data_catalog():
    # Read the data catalog file
    data_catalog = pd.read_csv('/Users/kellyliu/Documents/GitHub/Customer-Personality-Analysis-2.0/Data/Preprocessed Data/Final Preprocessed Data.csv')

    # Display the data catalog in a table
    st.table(data_catalog)

display_data_catalog()