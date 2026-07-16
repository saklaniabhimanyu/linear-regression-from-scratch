import numpy as np

from exceptions import ModelNotTrainedError
from metrics import r2_score, score

class BaseRegression:
    
    def __init__(self):
        self.weights = None
        self.bias = None
        
    def _check_is_fitted(self):
        if self.weights is None or self.bias is None:
            raise ModelNotTrainedError(
                'Model is not trained. Call fit() first'
            )
        
    def predict(self, X : np.ndarray)-> np.ndarray:
        '''
        Predict target values using the trained linear regression model
        Parameters
        X      : np.ndarray - Input feature matrix of shape (n_samples, n_features).
        Returns: np.ndarray - Predicted target values.
        '''
        self._check_is_fitted()
        return X @ self.weights + self.bias

    def evaluate(self, y_true : np.ndarray, y_pred : np.ndarray) -> dict:
        '''
        Evaluate Regression model using MSE, MAE, RMSE, MAPE, R2
        '''
        return score(y_true, y_pred)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        '''
        Return the coefficient of determination (R²)
        '''
        return r2_score(y, self.predict(X))
