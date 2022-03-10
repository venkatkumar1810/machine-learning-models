# Simple Linear regression 
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#  importing the dataset

dataset = pd.read_csv("Salary_data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Spliting the data set in the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 0)

# training the simple linear regression model on the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# predicting the test set results
y_pred = regressor.predict(X_test)

# Visulazing the training set results
plt.scatter(X_train, y_train, color="red")
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title("Salary Vs Experience (Training Set)")
plt.xlabel("years of Experience")
plt.ylabel("Salary")
plt.show()

# Visulaling the test set results
plt.scatter(X_test, y_test, color="red")
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title("Salary Vs Experience (Test Set)")
plt.xlabel("years of Experience")
plt.ylabel("Salary")
plt.show()


