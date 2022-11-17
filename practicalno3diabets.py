# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 20:39:15 2022

@author: LENOVO
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
df=pd.read_csv("diabetes.csv")
print(df.head())
df.shape
df.columns
df.isna().sum()
x=df.drop(["Outcome"],axis=1)
y=df["Outcome"]
x.shape
y.shape
x_train, x_test,y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=42)
knn= KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
