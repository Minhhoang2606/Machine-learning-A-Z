'''
Random Forest Regression
'''
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')

# Separating features and target variable
X = dataset.iloc[:, 1:2].values  # Level (independent variable)
y = dataset.iloc[:, 2].values  # Salary (dependent variable)

# Inspecting the dataset
print(dataset.head())

# Importing the Random Forest Regression model
from sklearn.ensemble import RandomForestRegressor

# Initializing the regressor
regressor = RandomForestRegressor(n_estimators=10, random_state=0)

# Making a prediction
y_pred = regressor.predict([[6.5]])

# Displaying the result
print(f"Predicted salary for level 6.5: ${y_pred[0]:,.2f}")


# Fitting the model to the dataset
regressor.fit(X, y)

# Predicting a new result
regressor.predict([[6.5]])

# Visualizing the Random Forest Regression results (high-resolution curve)
X_grid = np.arange(min(X), max(X), 0.01)  # Creates a grid with small step size
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color='red', label='Actual Data')  # Scatter plot of actual data
plt.plot(X_grid, regressor.predict(X_grid), color='blue', label='Model Prediction')  # Regression line
plt.title('Random Forest Regression Results')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.legend()
plt.show()

# Trying different values for n_estimators
for n in [10, 50, 100, 200]:
    regressor = RandomForestRegressor(n_estimators=n, random_state=0)
    regressor.fit(X, y)
    score = regressor.score(X, y)
    print(f"n_estimators={n}, R-squared Score: {score:.4f}")

from sklearn.model_selection import GridSearchCV

# Defining parameter grid
param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Performing Grid Search
grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=0),
                           param_grid=param_grid,
                           cv=5, scoring='r2')
grid_search.fit(X, y)

# Best parameters and score
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best R-squared Score: {grid_search.best_score_:.4f}")


