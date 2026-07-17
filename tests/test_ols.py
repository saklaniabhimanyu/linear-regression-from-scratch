import numpy as np
import pytest

import sys
sys.path.append("..")

from linear_regression.ols import LinearRegressionOLS
from linear_regression.exceptions import ModelNotTrainedError


def test_ols_initialization():
    model = LinearRegressionOLS()
    assert model.weights is None
    assert model.bias is None


def test_ols_fit():
    # y = 2x + 1
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
    model = LinearRegressionOLS()
    model.fit(X, y)
    assert model.weights is not None
    assert model.bias is not None


def test_ols_learns_correct_parameters():
    # y = 2x + 1
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
    model = LinearRegressionOLS()
    model.fit(X, y)
    np.testing.assert_allclose(
        model.weights,
        np.array([2]),
        rtol=1e-5
    )
    np.testing.assert_allclose(
        model.bias,
        1,
        rtol=1e-5
    )

def test_ols_prediction():
    X = np.array([
        [5],
        [6]
    ])

    y = np.array([
        11,
        13
    ])
    model = LinearRegressionOLS()
    model.fit(
        np.array([[1], [2], [3], [4]]),
        np.array([3,5,7,9])
    )
    predictions = model.predict(X)
    expected = np.array([11, 13])
    np.testing.assert_allclose(
        predictions,
        expected,
        rtol=1e-5
    )

def test_predict_before_fit():
    model = LinearRegressionOLS()
    X = np.array([[1]])
    with pytest.raises(ModelNotTrainedError):
        model.predict(X)
