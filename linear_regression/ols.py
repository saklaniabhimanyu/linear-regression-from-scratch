import numpy as np

from .base import BaseRegression

class LinearRegressionOLS(BaseRegression):
    '''
    Implements Linear Regression by using Oridnary Least Squares
                y_hat = X.w + b 
    Where :
        X - Input Features
        y - continous target variable
        w - weights/ coefficents
        b - bias/intercept
    Objective: min ||y - y_hat||^2
    Using Normal Equation: (X.T @ X) w = X.T @ y
    '''
    def __init__(self):
        super().__init__()

    def fit(self, X, y) :
        X = np.asarray(X)
        y = np.asarray(y)
        n, p = X.shape

        self.n_features_in_ = p
        #Add bias vector to the features
        X_b = np.column_stack([np.ones(n),X])
        #Compute w using Normal Equation
        w = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y

        self.bias = w[0]
        self.weights = w[1:]

        return self
