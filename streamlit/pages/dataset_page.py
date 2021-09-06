import streamlit as st
import sys
# setting path
sys.path.append('..\\azureml_sdk_utils')
from azureml_sdk_utils.azureml_sdk_utils import *

def app():
    st.title('Dataset')
    # st.markdown("## Original Dataset")

    ''' Upload the new dataset and save it to Azure '''
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
    
        st.markdown("## Uploaded Dataset")
        st.dataframe(df.head())
        st.button("Upload to Azure", on_click=upload_this_dataset)

    ''' Allowing the user to select a dataset from Azure Datastore '''
    all_registered_datasets = list(show_all_registered_datasets().keys())
    selected_dataset_name = st.selectbox("Choose a dataset from Azure Datastore", all_registered_datasets, index=0)
    selected_dataset_df = select_dataset(selected_dataset_name)
    st.dataframe(selected_dataset_df.head(n=20))

    # Getting and displaying the original dataset from AML
    # if view_original:
    #     loading_text = st.empty()
    #     loading_text.markdown('Fetching from Azure workspace...')
    #     # Fetching the dataset from AML takes quite a bit of time
    #     dataset_name = "KBest_new"
    #     original_dataset = select_dataset(dataset_name)
    #     # test = {'Name':['Tom', 'nick', 'krish', 'jack'],
    #     # 'Age':[20, 21, 19, 18]}
    #     # original_dataset = pd.DataFrame(test)
    #     st.dataframe(original_dataset.head(n=20))
    #     loading_text.markdown('')