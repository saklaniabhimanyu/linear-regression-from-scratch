import numpy as np
import pytest

import sys
sys.path.append("..")

from sklearn.linear_model import Ridge

from linear_regression.ridge import RidgeRegression
from linear_regression.exceptions import ModelNotTrainedError


def test_ridge_initialization():
    model = RidgeRegression()

    assert model.weights is None
    assert model.bias is None


def test_ridge_fit():
    X = np.array([
        [1],
        [2],
        [3],
        [4]
    ])

    y = np.array([
        3,
        5,
        7,
        9
    ])

    model = RidgeRegression(alpha=1.0)
    model.fit(X, y)

    assert model.weights is not None
    assert model.bias is not None


def test_ridge_matches_sklearn_parameters():
    X = np.array([
        [1],
        [2],
        [3],
        [4]
    ])

    y = np.array([
        3,
        5,
        7,
        9
    ])

    my_model = RidgeRegression(alpha=1.0)
    my_model.fit(X, y)

    sk_model = Ridge(alpha=1.0)
    sk_model.fit(X, y)

    np.testing.assert_allclose(
        my_model.weights,
        sk_model.coef_,
        rtol=1e-5
    )

    np.testing.assert_allclose(
        my_model.bias,
        sk_model.intercept_,
        rtol=1e-5
    )


def test_ridge_prediction_matches_sklearn():
    X_train = np.array([
        [1],
        [2],
        [3],
        [4]
    ])

    y_train = np.array([
        3,
        5,
        7,
        9
    ])

    X_test = np.array([
        [5],
        [6]
    ])

    my_model = RidgeRegression(alpha=1.0)
    my_model.fit(X_train, y_train)

    sk_model = Ridge(alpha=1.0)
    sk_model.fit(X_train, y_train)

    my_predictions = my_model.predict(X_test)
    sk_predictions = sk_model.predict(X_test)

    np.testing.assert_allclose(
        my_predictions,
        sk_predictions,
        rtol=1e-5
    )


def test_alpha_zero_matches_ols():
    X = np.array([
        [1],
        [2],
        [3],
        [4]
    ])

    y = np.array([
        3,
        5,
        7,
        9
    ])

    model = RidgeRegression(alpha=0.0)
    model.fit(X, y)

    np.testing.assert_allclose(
        model.weights,
        np.array([2.0]),
        rtol=1e-5
    )

    np.testing.assert_allclose(
        model.bias,
        1.0,
        rtol=1e-5
    )


def test_predict_before_fit():
    model = RidgeRegression()

    X = np.array([[1]])

    with pytest.raises(ModelNotTrainedError):
        model.predict(X)


def test_invalid_alpha():
    with pytest.raises(ValueError):
        RidgeRegression(alpha=-1)
