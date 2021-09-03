import streamlit as st
import pandas as pd
import numpy as np
import datetime

def app():
  st.title('Model Training')
  df=pd.read_csv("output.csv",parse_dates=True)
  date_columns = [column for column in df.columns if "DATE" in column]
  for date_column in date_columns:
    df[date_column] = pd.to_datetime(df[date_column], infer_datetime_format=True)
  df_names=df[["STOCK CODE","ISIN CODE","STOCK NAME","FACILITY CODE","ISSUER NAME","VALUE DATE","VALUE DATE MONTH"]]
  df.drop(["STOCK CODE","ISIN CODE","STOCK NAME","FACILITY CODE","ISSUER NAME","EXPECTED MATURITY DATE",'CONVERTIBLE/EXCHANGABLE',
          "FIRST PAYMENT DATE","NEXT PAYMENT DATE","PREVIOUS PAYMENT DATE","RATING EFFECTIVE DATE","ISSUE DATE","MATURITY DATE"],axis=1,inplace=True)
  continuous_df=df[['EVAL UPPER THRESHOLD YIELD', 'EVAL MID YIELD','EVAL LOWER THRESHOLD YIELD', 
                  'EVAL LOWER THRESHOLD PRICE', 'EVAL MID PRICE', 'EVAL UPPER THRESHOLD PRICE', 
                  'EVAL UPPER THRESHOLD YIELD CHANGE','EVAL MID YIELD CHANGE', 'EVAL LOWER THRESHOLD YIELD CHANGE',
                  'EVAL LOWER THRESHOLD PRICE CHANGE', 'EVAL MID PRICE CHANGE','EVAL UPPER THRESHOLD PRICE CHANGE', 
                  'MODIFIED DURATION','CONVEXITY', 'COMPOSITE LIQUIDITY SCORE (T-1)',
                  'COUPON FREQUENCY','PREVIOUS COUPON RATE', 'NEXT COUPON RATE',
                  'FACILITY AMOUNT/FACILITY LIMIT(MYR MIL)','FACILITY OUTSTANDING AMOUNT(MYR MIL)', 
                  'BOND ISSUE AMOUNT(MYR MIL)','BOND CURRENT OUTSTANDING AMOUNT(MYR MIL)', 
                  'ISSUER FACILITY LIMIT(MYR MIL)', 'ISSUER OUTSTANDING AMOUNT(MYR MIL)',
                   'MATURITY DURATION', 'ACCRUED INTEREST', 'OPR MOVEMENT', 'INFLATION RATE', 'MGS','CREDIT SPREAD', 
                  'CHANGES IN EVAL MID PRICE','CHANGES IN EVAL LOWER THRESHOLD PRICE','CHANGES IN EVAL UPPER THRESHOLD PRICE', 
                  'CHANGES IN EVAL MID YIELD','CHANGES IN EVAL LOWER THRESHOLD YIELD','CHANGES IN EVAL UPPER THRESHOLD YIELD',
                  'MA-5 CHANGES IN EVAL MID PRICE', 'MA-5 CHANGES IN EVAL MID YIELD']]
  nominal_df=df[['BOND TYPE', 'BOND CLASS','DAY COUNT BASIS','RATING AGENCY', 'ISLAMIC CONCEPT', 'SECTOR', 'RATING ACTION','RATING', 'REMAINING TENURE']]
  binary_df=df[['PRINCIPLE','CALLABLE/PUTTABLE']]
  target=df[['NEXT MONTH CHANGES IN EVAL MID PRICE']]

  #Mean normalization
  normalized_df=(continuous_df-continuous_df.mean())/continuous_df.std()
  #One hot encoding
  from sklearn.preprocessing import OneHotEncoder
  oh_encoder=OneHotEncoder(drop='first',sparse=False)
  encoded_nominal=pd.DataFrame(oh_encoder.fit_transform(nominal_df))
  encoded_nominal.columns=oh_encoder.get_feature_names()

  df_normalized=pd.DataFrame()
  df_normalized=pd.concat([normalized_df,encoded_nominal,binary_df,target],axis=1)
  from sklearn.feature_selection import SelectKBest
  from sklearn.feature_selection import f_classif
  from sklearn.feature_selection import RFE
  from sklearn.linear_model import LinearRegression
  from sklearn.ensemble import ExtraTreesRegressor

  def feature_selection(df_normalized,method,k, show_discard=False):
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
    X = df_normalized.drop(['NEXT MONTH CHANGES IN EVAL MID PRICE'],axis=1)
    Y = df_normalized[['NEXT MONTH CHANGES IN EVAL MID PRICE']]
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
      discarded=set(df_normalized.columns)-set(df_normalized.iloc[:,cols].columns)
      for discard in discarded:
        print(discard)
    df_normalized_new = df_normalized.iloc[:,cols]
    df_normalized_new=pd.concat([df_normalized_new,Y],axis=1)
    return df_normalized_new

  df_normalized_new=feature_selection(df_normalized,"K_Best",100,False)
  df_normalized_new=pd.concat([df_names[["STOCK CODE","VALUE DATE MONTH"]],df_normalized_new],axis=1)

  df_normalized_new=pd.concat([df_names[["VALUE DATE"]],df_normalized_new],axis=1)
  df_normalized_new.drop(["STOCK CODE","VALUE DATE MONTH"],axis=1,inplace=True)
  train=df_normalized_new.loc[df_normalized_new["VALUE DATE"]<=datetime.datetime(2020,7,31),:].reset_index(drop=True)
  test=df_normalized_new.loc[df_normalized_new["VALUE DATE"]>datetime.datetime(2020,7,31),:].reset_index(drop=True)
  y_train=train[["NEXT MONTH CHANGES IN EVAL MID PRICE"]]
  X_train=train.drop(["NEXT MONTH CHANGES IN EVAL MID PRICE","VALUE DATE"],axis=1)
  y_test=test[["NEXT MONTH CHANGES IN EVAL MID PRICE"]]
  X_test=test.drop(["NEXT MONTH CHANGES IN EVAL MID PRICE","VALUE DATE"],axis=1)

  from xgboost import XGBRegressor
  from sklearn.metrics import mean_squared_error
  
  # only for testing purpose
  model = XGBRegressor()
  eval_set = [(X_train, y_train),(X_test, y_test)]
  model.fit(X_train, y_train, eval_metric="error", eval_set=eval_set, verbose=False)
  # make predictions for test data
  y_pred = model.predict(X_test)
  predictions = [round(value) for value in y_pred]
  # evaluate predictions
  error = mean_squared_error(y_test, predictions)
  print("Mean squared error: {:.2f}".format(error))
  
  import matplotlib.pyplot as plt
  # retrieve performance metrics
  results = model.evals_result()
  epochs = len(results['validation_0']['error'])
  x_axis = range(0, epochs)
  # plot error
  fig, ax = plt.subplots()
  ax.plot(x_axis, results['validation_0']['error'], label='Train')
  ax.plot(x_axis, results['validation_1']['error'], label='Test')
  ax.legend()
  plt.ylabel('Mean Squared Error')
  plt.title('XGBoost Mean Squared Error')
  #pyplot.show()
  st.pyplot(fig)

  import shap
  import streamlit.components.v1 as components
  shap.initjs() # for visualization
  st.set_option('deprecation.showPyplotGlobalUse', False)
  def st_shap(plot, height=None):
      shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
      components.html(shap_html, height=height)

  explainer = shap.Explainer(model)
  shap_values = explainer(X_train)

  # visualize the first prediction's explanation
  shap.plots.waterfall(shap_values[0])
  #plt.savefig("waterfall.png",bbox_inches="tight")
  st.pyplot(bbox_inches='tight')

  options = st.selectbox(
       'SHAP options',
       ('Force','Summary', 'Beeswarm'))
  if options=="Force":
    # visualize the first prediction's explanation with a force plot
    st_shap(shap.plots.force(explainer.expected_value, shap_values.values[0,:], X_train.iloc[0,:]))
  elif options=="Summary":
    # summarize the effects of all the features
    shap.summary_plot(shap_values, X_train, plot_type="bar")
    st.pyplot(bbox_inches='tight')
    #plt.savefig("summary.png",bbox_inches="tight")
    #st.image('summary.png')
  else:
    shap.plots.beeswarm(shap_values)
    st.pyplot(bbox_inches='tight')
    #plt.savefig("beeswarm.png",bbox_inches="tight")
    #st.image('beeswarm.png')
