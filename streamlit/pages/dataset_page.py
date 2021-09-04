import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
  
# setting path
sys.path.append('..\\azureml_sdk_utils')
from azureml_sdk_utils.azureml_sdk_utils import *

def app():
    st.title('Dataset')
    st.markdown("## Original Dataset")
    view_original = st.button('View original dataset')
    # Getting and displaying the original dataset from AML
    if view_original:
        # Fetching the dataset from AML takes quite a bit of time
        dataset_name = "Extra_Trees_new"
        original_dataset = select_dataset(dataset_name)
        st.dataframe(original_dataset.head())

    # Upload the dataset and save it to Azure
    st.markdown("## Data Upload")
    st.markdown("### Upload a new dataset for training.") 
    st.write("\n")

    uploaded_file = st.file_uploader("Choose a file", type=["xlsx","csv"])
    if uploaded_file is not None:
        filename, file_extension = os.path.splitext(uploaded_file.name)
        if file_extension == "xlsx":
            df = pd.read_excel(uploaded_file,  engine='openpyxl')
        else:
            df = pd.read_csv(uploaded_file)
        def upload_this_dataset():
            upload_dataset(filename, df)
    
        st.markdown("## New Dataset")
        st.dataframe(df.head())
        st.button("Upload to Azure", on_click=upload_this_dataset)

    # For debugging purposes
    st.table(pd.DataFrame(list(Dataset.get_all(WS).keys()), columns=["Dataset"]))
    # for dataset_name, dataset_file in Dataset.get_all(WS).items():
