import numpy as np

from .base import BaseRegression
from .ols import LinearRegressionOLS
from .polynomial_features import PolynomialFeatures


class PolynomialRegression(BaseRegression):
    def __init__(self, degree=2):
        super().__init__()
        self.degree = degree
        self.poly = PolynomialFeatures(degree)
        self.model = LinearRegressionOLS()

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        X_poly = self.poly.transform(X)
        self.model.fit(X_poly, y)

        # Copy learned parameters to BaseRegression
        self.weights = self.model.weights
        self.bias = self.model.bias
        self.n_features_in_ = X_poly.shape[1]

        return self

    def predict(self, X):
        X = np.asarray(X)
        X_poly = self.poly.transform(X)
        return self.model.predict(X_poly)
