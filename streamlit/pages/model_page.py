from altair.vegalite.v4.schema.channels import X2Value
import streamlit as st
import altair as alt
import numpy as np
import pandas as pd
from azureml.core.run import Run
import re
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.workspace import Workspace
from azureml.core.experiment import Experiment
from azureml.train.automl.run import AutoMLRun
# import sys

# sys.path.append('..\\')

def fetch_newest(fetch=False,csv='/bond-price-forcasting/streamlit/pages/exp.csv'):
  if fetch:
    
    # get auth code
    SVC_PR = ServicePrincipalAuthentication(
    tenant_id="a63bb1a9-48c2-448b-8693-3317b00ca7fb",
    service_principal_id="70f9b97e-bc1f-4cad-a9f7-46cb12709d90",
    service_principal_password="eE.ozUaf8ncZt8~5gID-A7X85g.7clQ.P6")

    # define workspace
    # WS = Workspace.from_config("/bond-price-forcasting/streamlit/azureml_sdk_utils/config.json", auth=SVC_PR)
    WS = Workspace.from_config("azureml_sdk_utils/config.json", auth=SVC_PR)

    experiments=[]
    run_ids=[]
    experiment_list=Experiment.list(WS)
    for experiment in experiment_list:
      run_list=experiment.get_runs(include_children=False)
      for run in run_list:
        experiments.append(experiment.name)
        run_ids.append(run.id)
    model_dict={}
    for i in range(len(experiments)):
      current=Experiment(WS, experiments[i])
      run=Run(current, run_ids[i])
      automl_run = AutoMLRun(current, run_id = run_ids[i])
      try:
        best_run, fitted_model = automl_run.get_output(include_children=True)
        metric=best_run.get_metrics()
        if 'root_mean_squared_error' in metric.keys():
          #print(best_run.properties['pipeline_spec'])
          #Cannot convert to dict because of the way some metric is stored, dict in list in dict etc.
          model_names=re.findall('"class_name":\s*"(.{1,30})",\s*"module"',best_run.properties['pipeline_spec'])
          model_loss=metric['root_mean_squared_error']
          if len(model_names)!=0:
            model_name=""
            for model in model_names:
              model_name+="+"+model
            model_name=model_name[1:]
            #print(experiments[i])
            #print(model_name)
            #print(model_loss)
            if experiments[i] in model_dict.keys():
              if model_name in model_dict[experiments[i]].keys():
                if model_loss>model_dict[experiments[i]][model_name]:
                  continue
            else:
              model_dict[experiments[i]]={}
            model_dict[experiments[i]][model_name]=model_loss
      except:
        continue

    df=pd.DataFrame(columns={"Experiment","Model","Root Mean Squared Error"})
    for experiment, loss_dict in model_dict.items():
      for model, loss in loss_dict.items():
        df=pd.concat([df,pd.DataFrame({"Experiment":[experiment],"Model":[model],"Root Mean Squared Error":[loss]})],ignore_index=True,axis=0)
    df.sort_values("Root Mean Squared Error",ascending=True,inplace=True)
    df=df.groupby("Experiment").head(1).reset_index(drop=True)
  else:
    df=pd.read_csv(csv)
  return df

def app():
    st.title('Model')
    st.write('Best model by experiment')
    # df=fetch_newest(False,'exp.csv')
    df=fetch_newest(False,r'pages/exp.csv')
    # df=fetch_newest(False,r'streamlit/pages/exp.csv')
    st.write(df.head(10))
