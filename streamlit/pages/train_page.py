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

    ''' Allowing the user to select a dataset from Azure Datastore '''
    all_registered_datasets = show_all_registered_datasets()
    selected_dataset_name = st.selectbox("Choose a dataset from Azure Datastore", list(all_registered_datasets.keys()), index=0)
    selected_dataset_df = select_dataset(selected_dataset_name, all_registered_datasets)

    ''' Allowing the user to select features to train on '''
    with st.expander("Featurization"):
        # clicked = st.button("Auto Feature Selection")
        selected_features = []
        # if clicked:
        #     selected_features = auto_feature_selection(selected_dataset_df, "RFE", 10, "NEXT MONTH CHANGES IN EVAL MID PRICE", show_discard=False)
        checked_box = {}
        for i, col in enumerate(selected_dataset_df.columns):
            selected = True
            if i % 2 == 0:
                cols = st.columns(2)
            if col not in selected_features and len(selected_features) != 0:
                selected = False
            checked_box[col] = cols[i % 2].checkbox(col, value=selected, key=col)

    ''' Allowing the user to enter and select certain parameters '''
    with st.form(key='training_parameters'):
        experiment_name = st.text_input(label='Experiment name')
        time_column_name = st.selectbox("Time column name", selected_dataset_df.columns, index=0)
        time_series_id_column_names = st.selectbox("Time series id column name", selected_dataset_df.columns, index=0)
        target_column_name = st.selectbox("Target column name", selected_dataset_df.columns, index=0)
        experiment_timeout = st.slider('hours', min_value=1, max_value=24)
        submit_button = st.form_submit_button(label='Start training')

    ''' Show training process '''
    if submit_button:
        allowed_features = [feature_col for feature_col, allowed in checked_box.items() if allowed]
        tabular_dataset = select_dataset(selected_dataset_name, all_registered_datasets, to_pandas_dataframe=False)
        tabular_dataset = tabular_dataset.keep_columns(allowed_features)
        train_model(tabular_dataset, experiment_name, time_column_name, time_series_id_column_names, target_column_name, experiment_timeout_hours=experiment_timeout)

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

