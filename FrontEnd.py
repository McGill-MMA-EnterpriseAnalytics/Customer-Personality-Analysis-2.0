import streamlit as st
import pandas as pd

# CSS to emulate Apple's style
apple_style_css = """
    <style>
        html, body, [class*="css"] {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            color: #333;
            background-color: #f5f5f7;
        }
        .stApp {
            max-width: 800px;
            margin: auto;
        }
        header, .reportview-container .main footer {
            display: none;
        }
    </style>
"""

# Embed the CSS
st.markdown(apple_style_css, unsafe_allow_html=True)

st.write("""
# Customer-Personality-Analysis-2.0
""")

# Setup tabs
tab1, tab2 = st.tabs(["Customer Data Catalog", "Power BI Embed"])

# Tab 1: Display customer data
with tab1:
    st.header("Customer Data Catalog")
    # Example data
    data = pd.read_csv('https://raw.githubusercontent.com/McGill-MMA-EnterpriseAnalytics/Customer-Personality-Analysis-2.0/main/Data/Preprocessed%20Data/Final%20Preprocessed%20Data.csv?token=GHSAT0AAAAAACL3VCJB4QQSO53AD2M7BBPIZRGSKYA')
    df = pd.DataFrame(data)
    st.write(df)

# Tab 2: Power BI Embed
with tab2:
    st.header("Power BI Report")
    # Replace `your_powerbi_embed_url` with the actual URL of your Power BI report
    powerbi_embed_url = "https://app.powerbi.com/groups/me/reports/57eb479c-bab5-4078-981c-e312dd61e67a/ReportSection?experience=power-bi"
    st.components.v1.html(f"""
        <iframe width="100%" height="600" src="{powerbi_embed_url}" frameborder="0" allowFullScreen="true"></iframe>
    """, height=620)

# Run the Streamlit app by navigating to the folder containing this script and typing:
# streamlit run app.py
