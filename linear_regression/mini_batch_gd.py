import numpy as np

from .base import BaseRegression

class LinearRegressionMiniBatchGD(BaseRegression):
    '''
    Implements Mini Batch Gradient Descent Linear Regression
                y_hat = X.w + b
    Where
        X     - Input Features
        y_hat - continous target variable
        w     - weights/ coefficents
        b     - bias/intercept
    '''
    def __init__(self, lr = 0.01, max_iter = 10000, batch_size = 32, shuffle = True, random_state = None, verbose = None):
        if lr <= 0:
            raise ValueError("learning_rate must be greater than 0")
        if max_iter <= 0:
            raise ValueError("max_iter must be greater than 0")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        self.learning_rate = lr
        self.max_iter = max_iter
        self.verbose = verbose
        self.loss_history_ = []
        self.shuffle = shuffle
        self.random_state = random_state
        self.batch_size = batch_size
        super().__init__()

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n, p = X.shape

        #Initialize parameters
        self.n_features_in_ = p
        self.weights = np.zeros(p)
        self.bias = 0.0
        rng = np.random.default_rng(self.random_state)
         
        for itr in range(self.max_iter):

            if self.shuffle:
                idx = rng.permutation(n)
            else:
                idx = np.arange(n)

            for i in range(0,n, self.batch_size):

                batch_idx = idx[i : i+ self.batch_size]

                X_batch = X[batch_idx]
                y_batch = y[batch_idx]
                #Forward pass
                y_hat = X_batch @ self.weights + self.bias
                #prediction error
                error = y_batch - y_hat

                batch_n = len(X_batch)
                # Compute gradients of the MSE loss
                dw = -(2/batch_n) * (X_batch.T @ error)
                db = -(2/batch_n)* np.sum(error)
                #Update parameters using gradient descent
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

            y_pred =  self.predict(X)
            loss = np.mean((y - y_pred) ** 2)
            if not np.isfinite(loss):
                raise ValueError(
                        "Training diverged. Reduce learning rate or scale features."
                    )
            self.loss_history_.append(loss)

            if self.verbose and itr % self.verbose == 0:
                print(f"Iteration {itr}, Loss: {loss:.6f}")

        return self
