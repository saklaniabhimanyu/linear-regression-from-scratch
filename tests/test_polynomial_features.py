import numpy as np

import sys
sys.path.append("..")

from linear_regression.polynomial_features import PolynomialFeatures


def test_degree_2():

    X = np.array([
        [1],
        [2],
        [3]
    ])

    poly = PolynomialFeatures(degree=2)

    X_poly = poly.transform(X)

    expected = np.array([
        [1, 1],
        [2, 4],
        [3, 9]
    ])

    np.testing.assert_array_equal(X_poly, expected)


def test_degree_3():

    X = np.array([
        [1],
        [2],
        [3]
    ])

    poly = PolynomialFeatures(degree=3)

    X_poly = poly.transform(X)

    expected = np.array([
        [1, 1, 1],
        [2, 4, 8],
        [3, 9, 27]
    ])

    np.testing.assert_array_equal(X_poly, expected)


def test_multifeature_degree_2():

    X = np.array([
        [1, 2],
        [3, 4]
    ])

    poly = PolynomialFeatures(degree=2)

    X_poly = poly.transform(X)

    expected = np.array([
        [1, 2, 1, 4],
        [3, 4, 9, 16]
    ])

    np.testing.assert_array_equal(X_poly, expected)
