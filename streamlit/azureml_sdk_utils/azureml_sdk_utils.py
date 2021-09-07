import azureml.core
import pandas as pd
import logging
import os
import numpy as np
import queue
import _thread

from azureml.core.workspace import Workspace
from azureml.core.experiment import Experiment
from azureml.train.automl import AutoMLConfig
from azureml.automl.core.featurization import FeaturizationConfig
from azureml.automl.core.forecasting_parameters import ForecastingParameters
from azureml.core.dataset import Dataset
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.widgets import RunDetails
from azureml.core.run import Run
from azureml.core.model import Model
from azureml.train.automl.run import AutoMLRun

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

import streamlit as st

# get auth code
SVC_PR = ServicePrincipalAuthentication(
    tenant_id="a63bb1a9-48c2-448b-8693-3317b00ca7fb",
    service_principal_id="70f9b97e-bc1f-4cad-a9f7-46cb12709d90",
    service_principal_password="eE.ozUaf8ncZt8~5gID-A7X85g.7clQ.P6")

# define workspace
WS = Workspace.from_config("streamlit/azureml_sdk_utils/config.json", auth=SVC_PR)
# WS = Workspace.from_config("azureml_sdk_utils/config.json", auth=SVC_PR)

# define a compute_cluster
amlcompute_cluster_name = "GigaBITS-compute"
COMPUTE_TARGET = ComputeTarget(workspace=WS, name=amlcompute_cluster_name)

# upload dataset
def upload_dataset(filename, df):
    df.to_csv(filename +".csv")
    datastore = WS.get_default_datastore()
    datastore.upload_files(files = [filename +".csv"], target_path = 'dataset/', overwrite = True,show_progress = True)
    dataset = Dataset.Tabular.from_delimited_files(path=datastore.path('dataset/'+filename +".csv"))
    dataset = dataset.register(workspace=WS,
                                 name=filename,
                                 description='testing',
                                 create_new_version = True)

# show all dataset names
# @st.cache(suppress_st_warning=True, hash_funcs={queue.SimpleQueue: id})
# @st.cache()
def show_all_registered_datasets():
    return list(Dataset.get_all(WS).key())
    
# select a dataset and convert it to pandas dataframe
# @st.cache(suppress_st_warning=True, hash_funcs={queue.SimpleQueue: id})
# @st.cache()
def select_dataset(dataset_name, to_pandas_dataframe = True):
    dataset = Dataset.get_by_name(WS, dataset_name, version='latest')
    if to_pandas_dataframe:
        return dataset.to_pandas_dataframe().reset_index(drop=True)
    else:
        return dataset

# feature selection
def manual_feature_selection(dataset_df, dropped_columns):
    return dataset_df.drop(dropped_columns)

# modified from goy code  
def auto_feature_selection(dataset_df, method, k, target_column_name, show_discard=False):
    """
    df_normalized: df
    The normalized dataframe
    method: str
    Can be either K_Best, RFE or Extra_Trees
    k: int
    The number of features to be retained.
    show_discard: bool
    Whether to print out the discarded features.
    """
    X = dataset_df.drop(columns = [target_column_name])
    Y = dataset_df[[target_column_name]]
    if method=="K_Best":
        selector = SelectKBest(score_func=f_classif, k=k)
    elif method=="RFE":
        model = LinearRegression()
        selector = RFE(model, n_features_to_select=k)
    elif method=="Extra_Trees":
        selector = ExtraTreesRegressor(n_estimators=100)
    else:
        print("Selection method not available.")
        return df_normalized
    fit = selector.fit(X, Y)
    cols=[]
    if method=="Extra_Trees":
        feature_list=list(selector.feature_importances_)
        for i in range(len(feature_list)):
            if feature_list[i]>0.01/k:
                cols.append(i)
    else:
        cols = selector.get_support(indices=True)
    if show_discard:
        discarded=set(dataset_df.columns)-set(dataset_df.iloc[:,cols].columns)
        for discard in discarded:
            print(discard)

    return cols+[target_column_name]

# show a list of experiments
def show_all_experiments():
    return Experiment.list(WS)

# select an experiment
def select_experiment(experiment_name):
    return Experiment(WS, experiment_name)

# show a list of runs from an experiment
def show_all_runs(experiment):
    return experiment.get_runs(include_children=True)

# select the best run in an experiment
def select_best_run(experiment):
    return experiment.get_runs(include_children=False)

# show a list of registered models
def show_all_models():
    return Model.list(WS)

# select a registered model
def select_registered_model(model_name):
    return Model(WS, model_name)

# select the best model from an experiment
def select_best_model(experiment, run_id, metric="root_mean_squared_error"):
    automl_run = AutoMLRun(experiment, run_id = run_id)
    best_run, fitted_model = automl_run.get_output(metric=metric)
    print(fitted_model.steps)
    return fitted_model

# make an inference with a model
def make_inference(model, x_future, y_future):
    model.quantiles = [0.05,0.5, 0.9] # provides the confidence interval
    result = model.forecast_quantiles(x_future, y_future)
    return result

# evaluate test set with root mean squared error
def evaluate_test_set(model, test_dataset, target_column):
    test_dataset["VALUE DATE MONTH"] = test_dataset["VALUE DATE MONTH"].astype('datetime64[ns]')
    x_test = test_dataset.copy()
    y_test = x_test.pop(target_column).values
    label_query = y_test.copy().astype(np.float)
    label_query.fill(np.nan)
    y_pred_df = model.forecast(x_test, label_query)[1]
    y_pred_df.dropna(subset=["_automl_target_col"],inplace=True) # _automl_target_col is the y_pred
    result = y_pred_df.merge(test_dataset[["VALUE DATE MONTH","STOCK CODE","NEXT MONTH CHANGES IN EVAL MID PRICE"]], on=["VALUE DATE MONTH","STOCK CODE"])
    rmse = sqrt(mean_squared_error(result[target_column], result["_automl_target_col"]))
    return result, rmse

# train a model on the cloud only 
def train_model(dataset_df, experiment_name, time_column_name, time_series_id_column_names, target_column_name, experiment_timeout_hours=24):
    # create or retrieve experiment
    experiment = select_experiment(experiment_name)
    forecasting_parameters = ForecastingParameters(
        time_column_name=time_column_name,
        forecast_horizon=1, # forecast 1 month ahead only
        time_series_id_column_names=time_series_id_column_names,
        freq='M' # Set the forecast frequency to be monthly 
    )

    automl_config = AutoMLConfig(task='forecasting',
                                enable_dnn=True,
                                primary_metric='normalized_root_mean_squared_error',
                                experiment_timeout_hours=experiment_timeout_hours,
                                training_data=dataset_df,
                                label_column_name=target_column_name,
                                compute_target=COMPUTE_TARGET,
                                enable_early_stopping=True,
                                featurization="auto",
                                n_cross_validations=3,
                                verbosity=logging.INFO,
                                max_concurrent_iterations=2,
                                max_cores_per_iteration=-1,
                                forecasting_parameters=forecasting_parameters)
    
    remote_run = experiment.submit(automl_config, show_output=True)
    # widget to show
    # RunDetails(remote_run).show()
    # https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/automated-machine-learning/local-run-classification-credit-card-fraud/auto-ml-classification-credit-card-fraud-local.ipynb
    remote_run.wait_for_completion()
    return remote_run.get_output()


# select_best_model(select_experiment("benchmark_bond_price_forecasting"),"AutoML_be919576-4584-4c93-a89c-91d9b7626971")
# print(list(show_all_registered_datasets().keys()))
# print(type(select_dataset("train_dataset_06092021", to_pandas_dataframe = False)))
# print(show_all_experiments())