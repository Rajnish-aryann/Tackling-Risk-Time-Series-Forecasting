import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
import xgboost as c
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
lbl=LabelEncoder()
os.chdir("/home/prajjwal/Downloads/ISI_Q2")
train=pd.read_csv("TrainingData.csv")
del train[train.columns[0]]
train=train[train.YR!=1993]
A=train[train.BL=="A"]
B=train[train.BL=="B"]
C=train[train.BL=="C"]
A1=A[A.QR==1]
A2=A[A.QR==2]
A3=A[A.QR==3]
A4=A[A.QR==4]
B1=B[B.QR==1]
B2=B[B.QR==2]
B3=B[B.QR==3]
B4=B[B.QR==4]
C1=C[C.QR==1]
C2=C[C.QR==2]
C3=C[C.QR==3]
C4=C[C.QR==4]
A1=A1.reset_index(drop=True)
A2=A2.reset_index(drop=True)
A3=A3.reset_index(drop=True)
A4=A4.reset_index(drop=True)
B1=B1.reset_index(drop=True)
B2=B2.reset_index(drop=True)
B3=B3.reset_index(drop=True)
B4=B4.reset_index(drop=True)
C1=C1.reset_index(drop=True)
C2=C2.reset_index(drop=True)
C3=C3.reset_index(drop=True)
C4=C4.reset_index(drop=True)
scaler = StandardScaler()
import statsmodels.api as sm
t=[["AS","C1"],["DW","C2"],["LL","C3"],["ZY","C4"],["QJ","C5"]]
final=[]
d=[A1,B1,C1,A2,B2,C2,A3,B3,C3,A4,B4,C4]
scalar = StandardScaler()
sub=pd.DataFrame(columns=['a','b','c','d','e'])
for j in d:
    ans=[]
    for x in t:
        df = pd.DataFrame(columns=['a','b','c','d','e','f','a1','b1','c1','d1','e1','f1','g',])
        l=j[x]
        l1=l[x[0]]
        l2=l[x[1]]
        l1[24]=np.NAN
        for i in range(19): 
            df=df.append({'a':l1[i],'b':l1[i+1],'c':l1[i+2],'d':l1[i+3],'e':l1[i+4],'f':l1[i+5],'a1':l2[i],'b1':l2[i+1],'c1':l2[i+2],'d1':l2[i+3],'e1':l2[i+4],'f1':l2[i+5],'g':l1[i+6]},ignore_index=True)
        train=df.iloc[:18,:12]
        scalar.fit(train)
        train=scalar.transform(train)
        Y=df.iloc[:18,12]
        test=df.iloc[18:,:12]
        test=scalar.transform(test)
        re = sm.OLS(Y, train).fit()
        predictions = re.get_prediction(test)
        an=predictions.summary_frame(alpha=0.001)
        an=an["mean_ci_lower"][0]
        ans.append(an)
    final.append(ans)
    
for j in d:
    ans=[]
    for x in t:
        df = pd.DataFrame(columns=['a','b','c','d','e','f','a1','b1','c1','d1','e1','f1','g',])
        l=j[x]
        l1=l[x[0]]
        l2=l[x[1]]
        l1[24]=np.NAN
        for i in range(19): 
            df=df.append({'a':l1[i],'b':l1[i+1],'c':l1[i+2],'d':l1[i+3],'e':l1[i+4],'f':l1[i+5],'a1':l2[i],'b1':l2[i+1],'c1':l2[i+2],'d1':l2[i+3],'e1':l2[i+4],'f1':l2[i+5],'g':l1[i+6]},ignore_index=True)
        train=df.iloc[:18,:12]
        scalar.fit(train)
        train=scalar.transform(train)
        Y=df.iloc[:18,12]
        test=df.iloc[18:,:12]
        test=scalar.transform(test)
        re = sm.OLS(Y, train).fit()
        predictions = re.get_prediction(test)
        an=predictions.summary_frame(alpha=0.001)
        an=an["mean_ci_upper"][0]
        ans.append(an)
    final.append(ans)   
    
import csv
with open("sub2.csv","w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(final)