import numpy as np
import pytest

import sys
sys.path.append("..")

from linear_regression.polynomial_regression import PolynomialRegression
from linear_regression.exceptions import ModelNotTrainedError


def test_polynomial_initialization():
    model = PolynomialRegression(degree=2)

    assert model.weights is None
    assert model.bias is None
    assert model.degree == 2


def test_polynomial_fit():
    # y = x² + 1
    X = np.array([
        [1],
        [2],
        [3],
        [4]
    ])

    y = np.array([
        2,
        5,
        10,
        17
    ])

    model = PolynomialRegression(degree=2)
    model.fit(X, y)

    assert model.weights is not None
    assert model.bias is not None


def test_polynomial_learns_correct_parameters():
    # y = x² + 1

    X = np.array([
        [1],
        [2],
        [3],
        [4]
    ])

    y = np.array([
        2,
        5,
        10,
        17
    ])

    model = PolynomialRegression(degree=2)
    model.fit(X, y)

    # Model: y = 1 + 0*x + 1*x²
    np.testing.assert_allclose(
        model.bias,
        1,
        rtol=1e-5
    )

    np.testing.assert_allclose(
        model.weights,
        np.array([0, 1]),
        atol=1e-5
    )


def test_polynomial_prediction():

    X_train = np.array([
        [1],
        [2],
        [3],
        [4]
    ])

    y_train = np.array([
        2,
        5,
        10,
        17
    ])

    model = PolynomialRegression(degree=2)
    model.fit(X_train, y_train)

    X_test = np.array([
        [5],
        [6]
    ])

    predictions = model.predict(X_test)

    expected = np.array([
        26,
        37
    ])

    np.testing.assert_allclose(
        predictions,
        expected,
        rtol=1e-5
    )


def test_predict_before_fit():

    model = PolynomialRegression(degree=2)

    X = np.array([[1]])

    with pytest.raises(ModelNotTrainedError):
        model.predict(X)
