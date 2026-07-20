import numpy as np

class PolynomialFeatures:

    def __init__(self, degree):
        self.degree = degree

    def transform(self, X):

        X_poly = X.copy()

        for d in range(2, self.degree + 1):
            X_poly = np.column_stack((X_poly, X ** d))

        return X_poly
