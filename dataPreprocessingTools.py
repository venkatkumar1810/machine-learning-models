# Data Preprocessing Tools 
    # Importingg the libraries

# Data Preprocessing Tools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the Datasets

dataset = pd.read_csv('data.csv')
# create matrix of features and dependent variable vector
# features are the columns with which we;re going to predict the dependent variablel
# dependent var is which is going to be predicted

# the matrix of features
X = dataset.iloc[:, :-1].values # locate indexes and extract them
            # all the range, # all the columns excluding the last colum 
# the matrix of dependent variables
y = dataset.iloc[:, -1].values

print(X)

print(y)

# Taking care of the missing data
# sklearn - all ML
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

print(X)

# Encode Categorical data
#  this is to encode catergorial data and avoid the model to think 
#  One Hot encoding its imprtant if 3 values in data then we represent it in 3 cols
#  as binary vectors

# Encoding the independent variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(x))

print(X)

# Encoding the dependent variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit_transform(y)

print(y)

# Splitting the dataset into the Training set and Test set

# we need to apply feature scaling last because if we apply before split it'll 
# cause leakage of informatation from training to testset

# 4 sets X train test, y train y test
# for training x train and x test as inputs
# for testing x test anad y test as inputs

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 1) 
# Test size 0.2 means 20% of the data goes to testing set
# 80% to training set , 
# random_state is for which will decide the splitting of data into train and test indices in your case

print(X_train)

print(X_test)

print(y_train)

print(y_test)

# Feature scaling
# standardization and normalization
# *** Xstand = (x - mean(x))/standardDeviation(x)
# * Xnorm = (x - min(x))/(max(x) - min(x))

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])

print(X_train)

print(X_test)

