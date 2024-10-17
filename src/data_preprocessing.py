import pandas as pd
import numpy as np

def preprocess(filepath):
    df=pd.read_csv(filepath)
# drops all names columns 
    df=df.drop(columns=['isFlaggedFraud','nameOrig','nameDest','oldbalanceDest','newbalanceDest'])
#creates a new voolean column that indicates iif amount and old balance are the same
    df['extracted']=df['amount']==df['oldbalanceOrg']
    df['extracted']=df['extracted'].astype(int)
#encodes step and type where 1= fraud, 0.5 = may or may not be fraud, 0= not farud
    df['step'] = np.where((df['step']>= 50 ) & (df['step'] <= 120), 1, 0.5)

    df['type'] = df['type'].map({'PAYMENT':0, 'CASH_IN':0, 'DEBIT':0, 'CASH_OUT':0.5, 'TRANSFER':0.5})

    df=df.drop(columns=['oldbalanceOrg','newbalanceOrig','amount'])
#shuffles and returns data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df




