import streamlit as st
import os
import pandas as pd
import numpy as np
import sys
import time
import altair as alt
# setting path
sys.path.append('..\\azureml_sdk_utils')
from azureml_sdk_utils.azureml_sdk_utils import *

def app():
    st.title('Train New Model')

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
    
        st.markdown("## New Dataset")
        st.dataframe(df.head())
        st.button("Upload to Azure", on_click=upload_this_dataset)

    # For debugging purposes
    # st.table(pd.DataFrame(list(show_all_registered_datasets()), columns=["Dataset"]))
    # for dataset_name, dataset_file in Dataset.get_all(WS).items():

    ''' Allowing the user to enter and select certain parameters '''
    with st.form(key='training_parameters'):
        experiment_name = st.text_input(label='Experiment name')
        time_column_name = st.text_input(label='Time column name')
        time_series_id_column_names = st.text_input(label='Time series id column name')
        target_column_name = st.text_input(label='Target column name')
        experiment_timeout = st.slider('hours', min_value=1, max_value=24)
        submit_button = st.form_submit_button(label='Start training')

    ''' Show training process '''
    if submit_button:
        #train_model(df, experiment_name, time_column_name, time_series_id_column_names, target_column_name, experiment_timeout_hours=experiment_timeout)

        # Progress bar sample
        training_text = st.empty()
        training_text.text('Training...')
        bar = st.progress(0)
        for i in range(100):
            # Update the progress bar with each iteration.
            bar.progress(i + 1)
            time.sleep(0.1)
        training_text.text('Training completed!')
        

        ''' Show graphs of accuracy and loss '''
        st.markdown('## New Model')
        col3, col4 = st.columns(2)
        with col3:
            st.write('Accuracy')

            # Sample accuracy graph
            x = np.arange(1,100)
            source = pd.DataFrame({
            'x': x,
            'f(x)': np.log(x)*20
            })

            chart = alt.Chart(source).mark_line().encode(
                x='x',
                y='f(x)'
            )
            st.altair_chart(chart, use_container_width=True)
        with col4:
            st.write('Loss')

            # Sample accuracy graph
            x = np.arange(1,100)
            source = pd.DataFrame({
            'x': x,
            'f(x)': [1/i for i in x]
            })

            chart = alt.Chart(source).mark_line().encode(
                x='x',
                y='f(x)'
            )
            st.altair_chart(chart, use_container_width=True)