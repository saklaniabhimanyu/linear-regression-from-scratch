import numpy as np
import pytest

import sys
sys.path.append("..")

from sklearn.linear_model import Lasso

from linear_regression.lasso import LassoRegression
from linear_regression.exceptions import ModelNotTrainedError


def test_lasso_initialization():
    model = LassoRegression()

    assert model.weights is None
    assert model.bias is None


def test_lasso_fit():
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

    model = LassoRegression(alpha=0.1)
    model.fit(X, y)

    assert model.weights is not None
    assert model.bias is not None


def test_lasso_matches_sklearn_parameters():
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

    my_model = LassoRegression(
        alpha=0.1,
        learning_rate=0.01,
        n_iters=50000
    )
    my_model.fit(X, y)

    sk_model = Lasso(
        alpha=0.1,
        fit_intercept=True,
        max_iter=50000,
        tol=1e-6
    )
    sk_model.fit(X, y)

    np.testing.assert_allclose(
        my_model.weights,
        sk_model.coef_,
        atol=1e-2
    )

    np.testing.assert_allclose(
        my_model.bias,
        sk_model.intercept_,
        atol=1e-2
    )


def test_lasso_prediction_matches_sklearn():
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

    my_model = LassoRegression(
        alpha=0.1,
        learning_rate=0.01,
        n_iters=50000
    )
    my_model.fit(X_train, y_train)

    sk_model = Lasso(
        alpha=0.1,
        max_iter=50000,
        tol=1e-6
    )
    sk_model.fit(X_train, y_train)

    my_predictions = my_model.predict(X_test)
    sk_predictions = sk_model.predict(X_test)

    np.testing.assert_allclose(
        my_predictions,
        sk_predictions,
        atol=1e-2
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

    model = LassoRegression(
        alpha=0.0,
        learning_rate=0.01,
        n_iters=50000
    )
    model.fit(X, y)

    np.testing.assert_allclose(
        model.weights,
        np.array([2.0]),
        atol=1e-2
    )

    np.testing.assert_allclose(
        model.bias,
        1.0,
        atol=1e-2
    )


def test_predict_before_fit():
    model = LassoRegression()

    X = np.array([[1]])

    with pytest.raises(ModelNotTrainedError):
        model.predict(X)


def test_invalid_alpha():
    with pytest.raises(ValueError):
        LassoRegression(alpha=-1)
