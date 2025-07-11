# Linear Regression: Predicting Exam Scores

This project demonstrates a simple linear regression model using Python and scikit-learn to predict exam scores based on hours studied.

## 📊 Project Overview

Using a small dataset of hours studied and corresponding exam scores, we:

- Train a linear regression model
- Evaluate its performance
- Visualize the results
- Make a prediction for a new input

## 🧠 Key Concepts

- **Linear Regression**: A supervised machine learning algorithm for predicting continuous values.
- **Model Evaluation**: Using Mean Squared Error (MSE) and R² score.
- **Data Visualization**: Scatter plots and regression line using matplotlib.

## 🛠️ Libraries Used

- `numpy`
- `matplotlib`
- `scikit-learn`

## 📁 Dataset

The dataset is manually created within the code:

```python
hours_studied = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
exam_scores = np.array([55, 60, 65, 68, 70, 75, 79, 85, 88, 90])

