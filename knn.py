import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import metrics


df=pd.read_csv("C:/Users/DELL/Downloads/archive/emails.csv")
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

metrics.accuracy_score(y_test, y_pred)
metrics.mean_absolute_error(y_test, y_pred) #mean absolute error

metrics.mean_squared_error(y_test, y_pred) #mean squared error

np.sqrt(metrics.mean_squared_error(y_test, y_pred)) # root mean squared error

#SVM


svm=SVC(C=1)
svm.fit(x_train, y_train)
y_pred=svm.predict(x_test)
print(metrics.accuracy_score(y_test, y_pred))

