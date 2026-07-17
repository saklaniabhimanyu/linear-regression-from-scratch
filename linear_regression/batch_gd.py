import numpy as np

from .base import BaseRegression

class LinearRegressionBatchGD(BaseRegression):
    '''
    Implements Batch Gradient Descent Linear Regression
                y_hat = X.w + b
    Where
        X     - Input Features
        y_hat - continous target variable
        w     - weights/ coefficents
        b     - bias/intercept
    '''
    def __init__(self, lr = 0.01, max_iter = 10000, verbose = None):
        if lr <= 0:
            raise ValueError("learning_rate must be greater than 0")
        if max_iter <= 0:
            raise ValueError("max_iter must be greater than 0")
        self.learning_rate = lr
        self.max_iter = max_iter
        self.verbose = verbose
        self.loss_history_ = []
        super().__init__()

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n, p = X.shape

        #Initialize parameters
        self.n_features_in_ = X.shape[1]
        self.weights = np.zeros(p)
        self.bias = 0.0

        for itr in range(self.max_iter):

            #Forward pass
            y_hat = X @ self.weights + self.bias
            #prediction error
            error = y - y_hat
            
            # Compute gradients of the MSE loss
            dw = -(2/n)* (X.T @ error)
            db = -(2/n)* np.sum(error)

            #Update parameters using gradient descent
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            loss = np.mean(error ** 2)
            self.loss_history_.append(loss)

            if self.verbose and itr % self.verbose == 0:
                print(f"Iteration {itr}, Loss: {loss:.6f}")

        return self
