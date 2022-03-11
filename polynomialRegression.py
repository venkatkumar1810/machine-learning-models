# position  level  Salary
# predict the previous salary for some level
# find if the person is saying truth or bluff about his salary
# we wont split the dataset for this dataset

# Polynomial Regression

# Importing the libraries
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# imprting the dataset
dataset = pd.read_csv('dataset-name.csv')
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,:-1].values

# Training the linear Regression model on the whole dataset
from sklearn.linear_model import LinearRegression
linReg = LinearRegression()
linReg.fit(X, y)

# Training the Polynomial Regression model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
polyReg = PolynomialFeatures(degree= 2)
X_poly = polyReg.fit_transform(X)
linReg2 = LinearRegression()
linReg2.fit(X_poly, y)

# visualizing the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, linReg.predict(X), color = 'blue')
plt.title("Truth or Bluff (Linear Regression)")
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualizing the polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, linReg2.predict(polyReg.fit_transform(X)), color = 'blue')
plt.title("Truth or Bluff (Linear Regression)")
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
 

# Visualizing the polynomial Regression results (for higher and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X_grid, linReg2.predict(polyReg.fit_transform(X_grid)), color= 'blue')
plt.title("Truth or Bluff (Polynomial Regression)")
plt.xlabel("Position Level")
plt.ylabel("salary")
plt.show()

# Predicting a new result with LinearRegression
linReg.predict([[6.5]])

# Predicting a new result with Polynomial Regression
linReg2.predict(polyReg.fit_transform([[6.5]]))
