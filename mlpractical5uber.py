# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 18:44:24 2022

@author: LENOVO
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score,mean_absolute_error,mean_squared_error

 
df= pd.read_csv("C:/Users/LENOVO/Downloads/uber.csv")
df.head(10)


df.columns

df= df.drop(["Unnamed: 0", "key"], axis=1)
df.head()

df.isna().sum()

df= df.dropna(axis=0)
df.isna().sum()

df.shape

df.dtypes

df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
df.dtypes

df= df.assign(
    hour=df.pickup_datetime.dt.hour,
    day=df.pickup_datetime.dt.day,
    month=df.pickup_datetime.dt.month,
    year=df.pickup_datetime.dt.year,
    dayofweek=df.pickup_datetime.dt.dayofweek)

df.shape

df= df.drop("pickup_datetime", axis=1)
df.shape

x= df.drop(["fare_amount"], axis=1)
y=df["fare_amount"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# Linear REgression
model=LinearRegression()
model.fit(x_train, y_train)
y_pred=model.predict(x_test)

mean_absolute_error(y_test, y_pred) #mean absolute
mean_squared_error(y_test, y_pred) #mean squared error

np.sqrt(mean_squared_error(y_test, y_pred)) # root mean squared error

#random forest classifier


model1=RandomForestRegressor()
model1.fit(x_train, y_train)
y_pred=model1.predict(x_test)

mean_absolute_error(y_test, y_pred) #mean absolute error

mean_squared_error(y_test, y_pred) #mean squared error

np.sqrt(mean_squared_error(y_test, y_pred)) # root mean squared error


