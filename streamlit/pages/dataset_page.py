import streamlit as st
import sys
from azureml_sdk_utils.azureml_sdk_utils import *
import pandas as pd
import numpy as np
from pages import utils
  
# setting path
sys.path.append('..\\azureml_sdk_utils')

def app():
    st.title('Dataset')

    st.markdown("## Original Dataset")
    # Getting the original dataset from AML
    # df = Dataset.get_all(WS)["output_RFE"].to_pandas_dataframe()
    # st.dataframe(df.iloc[0].style.highlight_max(axis=0))
    # st.text('This will appear second')

    st.markdown("## Data Upload")

    # Upload the dataset and save as csv
    st.markdown("### Upload a new dataset for training.") 
    st.write("\n")

    # Code to read a single file and save it
    uploaded_file = st.file_uploader("Choose a file", type = ['csv', 'xlsx'])
    global data
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
        except Exception as e:
            print(e)
            data = pd.read_excel(uploaded_file)
        data.to_csv('uploaded_data/main_data.csv', index=False)

    if st.button("View New Dataset"):
        st.markdown("## New Dataset")
        # Raw data 
        st.dataframe(data)
        
