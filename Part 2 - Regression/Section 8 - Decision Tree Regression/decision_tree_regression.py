'''
Decision Tree Regression
Author: Henry Ha
'''
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor


# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
print(dataset.head())

X = dataset.iloc[:, 1:2].values  # Ensuring X is a 2D array
y = dataset.iloc[:, 2].values    # Target variable

print(X.shape, y.shape)


# Training the Decision Tree Regression model on the whole dataset
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict([[6.5]])
print(f"The predicted salary for position level 6.5 is: {y_pred[0]}")


# Visualising the Decision Tree Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()