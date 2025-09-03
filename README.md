# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
<img width="1017" height="403" alt="al ex2 ml" src="https://github.com/user-attachments/assets/a0edb177-3b70-4e4c-a919-d006e3795804" />

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Sivaprasath R
RegisterNumber: 25017023
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('student_scores.csv')
print(df.head())
print(df.tail())

# Features and target
X = df.iloc[:, :-1].values
print("X:", *X)
Y = df.iloc[:, 1].values
print("Y:", *Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_test)
print("Predicted:", *Y_pred)
print("Actual:", *Y_test)
plt.scatter(X_train, Y_train, color="orange")
plt.plot(X_train, regressor.predict(X_train), color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test, Y_test, color="blue")
plt.plot(X_test, regressor.predict(X_test), color="green")
plt.title("Hours vs Scores (Testing Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)

*/
```

## Output:
<img width="988" height="381" alt="ex2ML 1" src="https://github.com/user-attachments/assets/f913b43d-3e58-4330-a972-3278f9ef25cf" />

<img width="856" height="558" alt="ex2 ml 2" src="https://github.com/user-attachments/assets/bde9f7f8-3c88-4e0f-8b1d-c190eb5c30e5" />

<img width="946" height="653" alt="ex2 ml 3" src="https://github.com/user-attachments/assets/80a31245-d7e8-4168-8505-7225d42cb4ea" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
