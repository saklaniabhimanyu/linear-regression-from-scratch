import numpy as np
from .base import BaseRegression


class LassoRegression(BaseRegression):
    '''
    Implements Lasso Regression from scratch using Gradient Descent
    '''
    def __init__(self, alpha=1.0, learning_rate=0.01, n_iters=1000):
        if alpha < 0:
            raise ValueError("alpha must be non-negative")

        self.alpha = alpha
        self.learning_rate = learning_rate
        self.n_iters = n_iters

        super().__init__()

    def fit(self, X, y):
        n, p = X.shape

        # Initialize parameters
        self.weights = np.zeros(p)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.n_iters):

            # Predictions
            y_pred = X @ self.weights + self.bias

            # Errors
            error = y_pred - y

            # Gradient for weights
            dw = (1 / n) * (X.T @ error) + self.alpha * np.sign(self.weights)

            # Gradient for bias (no regularization)
            db = (1 / n) * np.sum(error)
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self
