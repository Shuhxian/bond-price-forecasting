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
  return df_normalized_new
