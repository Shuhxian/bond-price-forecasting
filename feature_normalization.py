from google.colab import files
#uploaded=files.upload()
df=pd.read_csv("output.csv")
df_names=df[["STOCK CODE","ISIN CODE","STOCK NAME","FACILITY CODE","ISSUER NAME","VALUE DATE MONTH"]]
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
df_normalized=pd.concat([df_names,normalized_df,encoded_nominal,binary_df,target],axis=1)
df_normalized
