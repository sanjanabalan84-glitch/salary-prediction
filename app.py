import streamlit as st
import pandas as pd

st.title("Indian Job Market Data")
st.write("Here's a look at the processed job market data:")

# Load the DataFrame from the CSV file
try:
    df_app = pd.read_csv('job_market_data.csv')
    st.dataframe(df_app)
except FileNotFoundError:
    st.error("Data file 'job_market_data.csv' not found. Please run the data saving step in Colab.")

st.write("Edit app.py to add more content!")
