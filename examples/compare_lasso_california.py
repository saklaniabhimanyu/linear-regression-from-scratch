import numpy as np
import pandas as pd
import sys
sys.path.append("..")

from linear_regression.lasso import LassoRegression 

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset California Housing
data = fetch_california_housing()

X = data.data
y = data.target

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=42)

#Standardize Data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Linear Regression Lasso Scratch 
my_model = LassoRegression()
my_model.fit(X_train, y_train)
my_model_pred = my_model.predict(X_test)
my_model_eval = my_model.evaluate(y_test, my_model_pred)

print("Lasso Regression from Scratch")
print(f"R2 Score : {my_model.score(X_test, y_test):.12f}")
print(f"MAE      : {my_model_eval['MAE']:.12f}")
print(f"MSE      : {my_model_eval['MSE']:.12f}")
print(f"RMSE     : {my_model_eval['RMSE']:.12f}")
print(f"MAPE (%) : {my_model_eval['MAPE']:.12f}")

# Scikit-learn Linear Regression
sk_model = Lasso()
sk_model.fit(X_train, y_train)
sk_pred = sk_model.predict(X_test)
sk_mae = mean_absolute_error(y_test, sk_pred)
sk_mse = mean_squared_error(y_test, sk_pred)
sk_rmse = np.sqrt(sk_mse)
sk_r2 = r2_score(y_test, sk_pred)
sk_mape = np.mean(np.abs((y_test - sk_pred) / y_test)) * 100


print("Scikit-learn Lasso Regression")
print(f"R2 Score : {sk_r2:.12f}")
print(f"MAE      : {sk_mae:.12f}")
print(f"MSE      : {sk_mse:.12f}")
print(f"RMSE     : {sk_rmse:.12f}")
print(f"MAPE (%) : {sk_mape:.12f}")

# Comparison Table
comparison = pd.DataFrame({
    "Metric": ["R2 Score", "MAE", "MSE", "RMSE", "MAPE (%)"],
    "Scratch Lasso Regression": [
        my_model.score(X_test, y_test),
        my_model_eval["MAE"],
        my_model_eval["MSE"],
        my_model_eval["RMSE"],
        my_model_eval["MAPE"]
    ],
    "Scikit-learn Lasso": [
        sk_r2,
        sk_mae,
        sk_mse,
        sk_rmse,
        sk_mape
    ]
})

comparison["Absolute Difference"] = (
    comparison["Scratch Lasso Regression"] - comparison["Scikit-learn Lasso"]
).abs()

print("Performance Comparison")
print(comparison.to_string(index=False))
