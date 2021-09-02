import azureml.core
import pandas as pd
import logging
import os

from azureml.core.workspace import Workspace
from azureml.core.experiment import Experiment
from azureml.train.automl import AutoMLConfig
from azureml.automl.core.featurization import FeaturizationConfig
from azureml.automl.core.forecasting_parameters import ForecastingParameters
from azureml.core.dataset import Dataset
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.widgets import RunDetails

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor

# get auth code
SVC_PR = ServicePrincipalAuthentication(
    tenant_id="a63bb1a9-48c2-448b-8693-3317b00ca7fb",
    service_principal_id="70f9b97e-bc1f-4cad-a9f7-46cb12709d90",
    service_principal_password="eE.ozUaf8ncZt8~5gID-A7X85g.7clQ.P6")

# define workspace
WS = Workspace.from_config("config.json", auth=SVC_PR)

# define a compute_cluster
amlcompute_cluster_name = "GigaBITS-compute"
COMPUTE_TARGET = ComputeTarget(workspace=WS, name=amlcompute_cluster_name)

# upload dataset
def upload_dataset(excel_file):
    filename, file_extension = os.path.splitext(excel_file)
    # convert excel to csv
    if file_extension == ".xlsx":
        pd.read_excel(excel_file,  engine='openpyxl').to_csv(filename +".csv")

    datastore = WS.get_default_datastore()
    datastore.upload_files(files = [filename +".csv"], target_path = 'dataset/', overwrite = True,show_progress = True)
    dataset = Dataset.Tabular.from_delimited_files(path=datastore.path('dataset/'+filename +".csv"))
    dataset = dataset.register(workspace=WS,
                                 name='dataset',
                                 description='testing',
                                 create_new_version = True)

# show all dataset
def show_all_registered_datasets():
    return Dataset.get_all(WS)
    
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
    X = dataset_df.drop([target_column_name])
    Y = dataset_df[[target_column_name]]
    if method=="K_Best":
        selector = SelectKBest(score_func=f_classif, k=k)
    elif method=="RFE":
        model = LinearRegression()
        selector = RFE(model, k=k)
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

    return dataset_df.iloc[:,cols+[target_column_name]]

# train a model
def train_model(dataset_name, experiment_name, time_column_name, time_series_id_column_names, target_column_name, experiment_timeout_hours=24):
    # create or retrieve experiment
    experiment = Experiment(WS, experiment_name)
    # get a registered dataset
    dataset = Dataset.get_by_name(WS, dataset_name, version='latest')
    forecasting_parameters = ForecastingParameters(
        time_column_name=time_column_name,
        forecast_horizon=1, # forecast 1 month ahead only
        time_series_id_column_names=time_series_id_column_names,
        freq='M' # Set the forecast frequency to be monthly 
    )

    automl_config = AutoMLConfig(task='forecasting',
                                primary_metric='normalized_root_mean_squared_error',
                                experiment_timeout_hours=experiment_timeout_hours,
                                training_data=dataset,
                                label_column_name=target_column_name,
                                compute_target=COMPUTE_TARGET,
                                enable_early_stopping=True,
                                featurization="auto",
                                n_cross_validations=5,
                                verbosity=logging.INFO,
                                max_cores_per_iteration=-1,
                                forecasting_parameters=forecasting_parameters)
    
    remote_run = experiment.submit(automl_config, show_output=True)
    # widget to show
    # RunDetails(remote_run).show()
    # https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/automated-machine-learning/local-run-classification-credit-card-fraud/auto-ml-classification-credit-card-fraud-local.ipynb
    remote_run.wait_for_completion()
    return remote_run.get_output()



# upload_dataset("dataset_31082021.xlsx")
# print(show_all_registered_datasets())
best_run, fitted_model = train_model("dataset", "test", "VALUE DATE MONTH", "STOCK CODE", "NEXT MONTH CHANGES IN EVAL MID PRICE")