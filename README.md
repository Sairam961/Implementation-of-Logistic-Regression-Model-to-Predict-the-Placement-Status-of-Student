# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: R.Sairam
RegisterNumber:  25000694
*/
```
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

df= pd.read_csv("Placement_Data.csv")  

X = df[['ssc_p','hsc_p','degree_p','etest_p','mba_p']]

y = df['status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("Predicted Placement Status:", y_pred)

plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual Status')

plt.scatter(range(len(y_test)), y_pred, color='red', marker='x', label='Predicted Status')

plt.title("Actual vs Predicted Placement Status")

plt.xlabel("Test Sample")

plt.ylabel("Placement Status (0=Not Placed, 1=Placed)")

plt.legend()

plt.show()


## Output:
<img src="ex5 output 1.png" alt="Output" width="500">

<img src="ex5 output 2.png" alt="Output" width="500">


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
