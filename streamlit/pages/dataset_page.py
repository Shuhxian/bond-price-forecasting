import streamlit as st
import sys
import os
import pandas as pd

# setting path
sys.path.append('..\\azureml_sdk_utils')
from azureml_sdk_utils.azureml_sdk_utils import *

def app():
    st.title('Dataset')
    uploaded_file = st.file_uploader("Choose a file", type=["xlsx","csv"])
    if uploaded_file is not None:
        filename, file_extension = os.path.splitext(uploaded_file.name)
        if file_extension == "xlsx":
            df = pd.read_excel(uploaded_file,  engine='openpyxl')
        else:
            df = pd.read_csv(uploaded_file)
        def upload_this_dataset():
            upload_dataset(filename, df)
    
        st.button("Upload to Azure", on_click=upload_this_dataset)
        st.dataframe(df.head())
    
    st.table(pd.DataFrame(list(Dataset.get_all(WS).keys()), columns=["Dataset"]))
    # for dataset_name, dataset_file in Dataset.get_all(WS).items():

    # df = Dataset.get_all(WS)["output_RFE"].to_pandas_dataframe()
    # st.dataframe(df.iloc[0].style.highlight_max(axis=0))
    # st.text('This will appear second')
