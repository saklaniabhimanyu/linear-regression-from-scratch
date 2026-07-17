import numpy as np

from .base import BaseRegression

class LinearRegressionSGD(BaseRegression):
    '''
    Implements Stochastic Gradient Descent Linear Regression
                y_hat = X.w + b
    Where
        X     - Input Features
        y_hat - continous target variable
        w     - weights/ coefficents
        b     - bias/intercept
    '''
    def __init__(self, lr = 0.01, max_iter = 10000, shuffle = True, random_state = None, verbose = None):
        self.learning_rate = lr
        self.max_iter = max_iter
        self.verbose = verbose
        self.loss_history_ = []
        self.shuffle = shuffle
        self.random_state = random_state
        super().__init__()

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n, p = X.shape

        #Initialize parameters
        self.n_features_in_ = X.shape[1]
        self.weights = np.zeros(p)
        self.bias = 0.0
        rng = np.random.default_rng(self.random_state)
         
        for itr in range(self.max_iter):

            if self.shuffle:
                idx = rng.permutation(n)
            else:
                idx = np.arange(n)

            for i in idx:

                X_i = X[i]
                y_i = y[i]
                #Forward pass
                y_hat = X_i @ self.weights + self.bias
                #prediction error
                error = y_i - y_hat
            
                # Compute gradients of the MSE loss
                dw = -2* X_i * error
                db = -2* error
                #Update parameters using gradient descent
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

            y_pred = X @ self.weights + self.bias
            loss = np.mean((y - y_pred) ** 2)
            if not np.isfinite(loss):
                raise ValueError(
                        "Training diverged. Reduce learning rate or scale features."
                    )
            self.loss_history_.append(loss)

            if self.verbose and itr % self.verbose == 0:
                print(f"Iteration {itr}, Loss: {loss:.6f}")

        return self
