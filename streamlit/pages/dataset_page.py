import streamlit as st
import sys
  
# setting path
sys.path.append('..\\azureml_sdk_utils')
from azureml_sdk_utils.azureml_sdk_utils import *

def app():
    st.title('Dataset')
    # df = Dataset.get_all(WS)["output_RFE"].to_pandas_dataframe()
    # st.dataframe(df.iloc[0].style.highlight_max(axis=0))
    # st.text('This will appear second')
