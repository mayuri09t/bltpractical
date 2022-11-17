# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 19:55:28 2022

@author: LENOVO
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error,accuracy_score
from sklearn.svm import SVC
from sklearn import metrics


df=pd.read_csv("C:/Users/LENOVO/Downloads/emails.csv")
print(df.head())
print(df.shape)

df.isna().sum()

df.drop(["Email No."], axis=1, inplace=True)
x=df.drop(["Prediction"], axis=1)
y=df["Prediction"]

print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=45)

knn =KNeighborsClassifier(n_neighbors=5)

knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)

mse=mean_squared_error(y_test,y_pred)
rmse=mean_squared_error(y_test,y_pred,squared=False)
ac=accuracy_score(y_test, y_pred)
print("/n")
print("KNN RESULT")
print(F'Accuracy:){ac}')
print(f'mse:{mse}')
print(f'rmse:{rmse}')

model_SVC =SVC(C=1)
model_SVC.fit(x_train,y_train)
y_pred_SVC =model_SVC.predict(x_test)
from sklearn.metrics import mean_squared_error,accuracy_score
mse=mean_squared_error(y_test,y_pred_SVC)
rmse=mean_squared_error(y_test, y_pred_SVC,squared=False)
ac=accuracy_score(y_test,y_pred_SVC)
print("/n")
print("SVM RESULT")
print(F'Accuracy:){ac}')
print(f'mse:{mse}')
print(f'rmse:{rmse}')



