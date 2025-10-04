import numpy as np

from data_preparing_and_visualization import X, y

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# # Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                    test_size=0.2, random_state=42)

model = LinearRegression()    # Initialize the Linear Regression model

# Reshape the feature arrays to 2D
X_train = np.array(X_train).reshape(-1, 1)
X_test = np.array(X_test).reshape(-1, 1)

model.fit(X_train, y_train)    # Train the model on the training data

y_predicted = model.predict(X_test)    # Make predictions on the test set

# Getting standard regression metrics
mae = mean_absolute_error(y_test, y_predicted)
mse = mean_squared_error(y_test, y_predicted)
r2_score = r2_score(y_test, y_predicted)

# Predict future activity for the next years
X_future = np.array([2023, 2024, 2025, 2026, 2027, 2028]).reshape(-1, 1)
y_future_predicted = model.predict(X_future)

score = model.score(X_test, y_test)    # Calculate R² score on the test set

print(f'Predicted activity on next years: {y_future_predicted}')
print(f'R² score: {score}')

print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {np.sqrt(mse)}')