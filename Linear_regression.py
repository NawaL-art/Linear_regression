# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Generate sample data (hours studied vs. exam score)
hours_studied = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
exam_scores = np.array([55, 60, 65, 68, 70, 75, 79, 85, 88, 90])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    hours_studied, exam_scores, test_size=0.3, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print model parameters and performance metrics
print(f"Model slope (coefficient): {model.coef_[0]:.2f}")
print(f"Model intercept: {model.intercept_:.2f}")
print(f"Mean squared error: {mse:.2f}")
print(f"R² score: {r2:.2f}")

# Create a figure for visualization
plt.figure(figsize=(10, 6))

# Plot training data
plt.scatter(X_train, y_train, color='blue', label='Training data')

# Plot testing data
plt.scatter(X_test, y_test, color='green', label='Testing data')

# Plot predictions
plt.scatter(X_test, y_pred, color='red', marker='x', s=100, label='Predictions')

# Plot regression line
x_range = np.linspace(0, 11, 100).reshape(-1, 1)
y_range_pred = model.predict(x_range)
plt.plot(x_range, y_range_pred, color='red', label='Regression line')

# Add labels and title
plt.title('Hours Studied vs. Exam Score: Linear Regression Model')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.grid(True, alpha=0.3)
plt.legend()

# Show plot
plt.show()

# Example prediction for 5.5 hours of study
new_hours = np.array([[5.5]])
predicted_score = model.predict(new_hours)
print(f"Predicted score for {new_hours[0][0]} hours of study: {predicted_score[0]:.2f}")