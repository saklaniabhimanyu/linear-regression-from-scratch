import numpy as np

from .base import BaseRegression

class RidgeRegression(BaseRegression):
    '''
    Implements Ridge regression from scratch
    '''
    def __init__(self,alpha = 1.0):
        if alpha < 0:
            raise ValueError(
                'alpha must be non-negative'
            )
        self.alpha = alpha
        super().__init__()

    def fit(self,X,y):
        n, p = X.shape
        X_b = np.column_stack([np.ones(n), X])
        # Add regularisation to all weights EXCEPT bias (index 0)
        reg_matrix = self.alpha * np.eye(p+1)
        reg_matrix[0, 0] = 0           # don't regularise the bias term

        self.w_full = np.linalg.inv(X_b.T@X_b + reg_matrix) @ X_b.T @ y

        self.bias = self.w_full[0];
        self.weights = self.w_full[1:]

        return self
