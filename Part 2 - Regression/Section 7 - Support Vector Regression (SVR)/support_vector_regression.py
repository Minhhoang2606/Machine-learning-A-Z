'''
Support Vector Regression
Author: Henry Ha
'''
# Support Vector Regression (SVR)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')

# Display the first few rows
print(dataset.head())

# Extract features (Position Level) and target variable (Salary)
X = dataset.iloc[:, 1:2].values  # Position level
y = dataset.iloc[:, 2].values   # Salary
print(X)
print(y)

# Create scalers for X and y
sc_X = StandardScaler()
sc_y = StandardScaler()

# Scale X and y
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1, 1))  # Reshape y to a 2D array

print("Scaled X:\n", X)
print("Scaled y:\n", y)

#TODO: Building the SVR Model

# Create the SVR model with RBF kernel
regressor = SVR(kernel='rbf')

# Train the SVR model on the scaled dataset
regressor.fit(X, y.ravel())  # Flatten y for compatibility

#TODO: Making Predictions

# Scale the input level 6.5
scaled_level = sc_X.transform([[6.5]])

# Make the prediction in scaled form
scaled_prediction = regressor.predict(scaled_level)

# Inverse transform the prediction to get the original salary scale
predicted_salary = sc_y.inverse_transform(scaled_prediction.reshape(-1, 1))

print(f"Predicted salary for position level 6.5: ${predicted_salary[0][0]:,.2f}")

# Inverse transform X and y for visualization
X_original = sc_X.inverse_transform(X)
y_original = sc_y.inverse_transform(y.reshape(-1, 1))

# Plot the SVR predictions
plt.scatter(X_original, y_original, color='red', label='Actual Salaries')
plt.plot(X_original, sc_y.inverse_transform(regressor.predict(X).reshape(-1, 1)), color='blue', label='SVR Predictions')
plt.title('SVR Results')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.legend()
plt.show()

# Generate high-resolution data for X
X_grid = np.arange(min(X_original), max(X_original), 0.01).reshape(-1, 1)

# Scale the high-resolution data and predict using the SVR model
X_grid_scaled = sc_X.transform(X_grid)
y_grid_pred = sc_y.inverse_transform(regressor.predict(X_grid_scaled).reshape(-1, 1))

# Plot the high-resolution results
plt.scatter(X_original, y_original, color='red', label='Actual Salaries')
plt.plot(X_grid, y_grid_pred, color='blue', label='SVR Predictions (High-Res)')
plt.title('SVR Results (High Resolution)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.legend()
plt.show()

#TODO: Comparing SVR with Polynomial Regression

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Create polynomial features (degree 4 is used for this example)
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X_original)

# Fit the polynomial regression model
poly_regressor = LinearRegression()
poly_regressor.fit(X_poly, y_original)

# Predict using the polynomial regression model
y_poly_pred = poly_regressor.predict(X_poly)

# Plot the Polynomial Regression results
plt.scatter(X_original, y_original, color='red', label='Actual Salaries')
plt.plot(X_original, y_poly_pred, color='blue', label='Polynomial Regression Predictions')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.legend()
plt.show()

