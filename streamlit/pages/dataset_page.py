import streamlit as st
import sys
# setting path
sys.path.append('..\\azureml_sdk_utils')
from azureml_sdk_utils.azureml_sdk_utils import *

def app():
    st.title('Dataset')
    st.markdown("## Original Dataset")
    view_original = st.button('View original dataset')
    # Getting and displaying the original dataset from AML
    if view_original:
        loading_text = st.empty()
        loading_text.markdown('Fetching from Azure workspace...')
        # Fetching the dataset from AML takes quite a bit of time
        dataset_name = "KBest_new"
        original_dataset = select_dataset(dataset_name)
        # test = {'Name':['Tom', 'nick', 'krish', 'jack'],
        # 'Age':[20, 21, 19, 18]}
        # original_dataset = pd.DataFrame(test)
        st.dataframe(original_dataset.head(n=20))
        loading_text.markdown('')