'''
Multiple Linear Regression
Author: Henry Ha
'''

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')

# Handle missing data
print(dataset.isnull().sum())

# Encoding the State column
dataset = pd.get_dummies(dataset, columns=['State'], drop_first=True)

# Splitting the dataset
X = dataset.drop('Profit', axis=1)
y = dataset['Profit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Multiple Linear Regression model on the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# View the coefficients and intercept
print("Coefficients:", regressor.coef_)
print("Intercept:", regressor.intercept_)


# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Plotting real vs predicted profits
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual Profits')
plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted Profits')
plt.plot(range(len(y_test)), y_test, color='blue', linestyle='dashed', alpha=0.5)
plt.plot(range(len(y_pred)), y_pred, color='red', linestyle='dashed', alpha=0.5)
plt.title('Actual vs Predicted Profits')
plt.xlabel('Startups (Index)')
plt.ylabel('Profit')
plt.legend()
plt.show()

np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))