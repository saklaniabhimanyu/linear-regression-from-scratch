import numpy as np
import pandas as pd
import sys
sys.path.append("..")

from linear_regression.polynomial_regression import PolynomialRegression

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = fetch_california_housing()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

my_model = PolynomialRegression(degree=2)
my_model.fit(X_train, y_train)
my_model_pred = my_model.predict(X_test)
my_model_eval = my_model.evaluate(y_test, my_model_pred)

print("Polynomial Regression from Scratch")
print(f"R2 Score : {my_model.score(X_test, y_test):.12f}")
print(f"MAE      : {my_model_eval['MAE']:.12f}")
print(f"MSE      : {my_model_eval['MSE']:.12f}")
print(f"RMSE     : {my_model_eval['RMSE']:.12f}")
print(f"MAPE (%) : {my_model_eval['MAPE']:.12f}")

sk_model = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("linear", LinearRegression())
])

sk_model.fit(X_train, y_train)

sk_pred = sk_model.predict(X_test)
sk_mae = mean_absolute_error(y_test, sk_pred)
sk_mse = mean_squared_error(y_test, sk_pred)
sk_rmse = np.sqrt(sk_mse)
sk_r2 = r2_score(y_test, sk_pred)
sk_mape = np.mean(np.abs((y_test - sk_pred) / y_test)) * 100

print("Scikit-learn Polynomial Regression")
print(f"R2 Score : {sk_r2:.12f}")
print(f"MAE      : {sk_mae:.12f}")
print(f"MSE      : {sk_mse:.12f}")
print(f"RMSE     : {sk_rmse:.12f}")
print(f"MAPE (%) : {sk_mape:.12f}")

comparison = pd.DataFrame({
    "Metric": ["R2 Score", "MAE", "MSE", "RMSE", "MAPE (%)"],
    "Scratch Polynomial Regression": [
        my_model.score(X_test, y_test),
        my_model_eval["MAE"],
        my_model_eval["MSE"],
        my_model_eval["RMSE"],
        my_model_eval["MAPE"]
    ],
    "Scikit-learn": [
        sk_r2,
        sk_mae,
        sk_mse,
        sk_rmse,
        sk_mape
    ]
})

comparison["Absolute Difference"] = (
    comparison["Scratch Polynomial Regression"] - comparison["Scikit-learn"]
).abs()

print("Performance Comparison")
print(comparison.to_string(index=False))
