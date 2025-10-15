import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from data_preparing_and_visualization import X, y

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

# Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
score = model.score(X_test, y_test)

print(f'MAE: {mae:.2f}, \n MSE: {mse:.2f},\n RMSE: {rmse:.2f},\n R²: {r2:.3f}')
print(f'R² score using model.score(): {score:.3f}')

# Future prediction
X_future = np.array([2023, 2024, 2025, 2026, 2027, 2028]).reshape(-1, 1)
y_future_pred = model.predict(X_future)
y_future_pred = np.maximum(y_future_pred, 0)  # avoid negative numbers

print(f'Predicted activity for future years: {y_future_pred}')

# Save model to .pkl file
with open('linear_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Visualization
plt.style.use('ggplot')
plt.figure(figsize=(10,6))
plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, model.predict(X), color='red', label='Linear fit')
plt.plot(X_future, y_future_pred, '--', color='green', label='Future prediction')

plt.xlabel('Year')
plt.ylabel('Number of comments')
plt.title('Reddit Comment Activity Trend & Prediction')
plt.legend()

plt.show()