import streamlit as st
import pandas as pd

st.title("ML Debugger")
file = st.file_uploader("Upload your dataset (CSV only): ",type = ["csv"])

#data preview
if file:
    df = pd.read_csv(file)

    st.subheader("Data Preivew")
    st.write(df.head())

#basic info function
from utils.analysis import basic_info
if file :
    info = basic_info(df)
    st.subheader("Basic Info")
    st.write(f"Rows: {info['rows']}")
    st.write(f"Columns: {info['columns']}")
    st.write("Column Names:", info["Column_list"])

from utils.analysis import missing_percent
if file:
    missing_per_val = missing_percent(df)
    st.subheader("Percentage of missing values per column")
    st.write(missing_per_val)

from utils.issues import max_class
if file:
    max_imb_value = max_class(df)
    st.warning(max_imb_value)


from utils.analysis import class_coverage
if file:
    value = class_coverage(df)
    st.subheader("Class Imbalance Analysis")
    st.write(value)
