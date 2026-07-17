import numpy as np
import pytest

from linear_regression.batch_gd import LinearRegressionBatchGD
from linear_regression.exceptions import ModelNotTrainedError


def test_batch_gd_initialization():
    model = LinearRegressionBatchGD()

    assert model.weights is None
    assert model.bias is None


def test_batch_gd_fit():
    X = np.array([[1], [2], [3], [4]])
    y = np.array([3, 5, 7, 9])  # y = 2x + 1

    model = LinearRegressionBatchGD(
        lr=0.1,
        max_iter=5000
    )

    model.fit(X, y)

    assert model.weights is not None
    assert model.bias is not None


def test_batch_gd_learns_correct_parameters():
    X = np.array([[1], [2], [3], [4]])
    y = np.array([3, 5, 7, 9])

    model = LinearRegressionBatchGD(
        lr=0.1,
        max_iter=5000
    )

    model.fit(X, y)

    np.testing.assert_allclose(
        model.weights,
        np.array([2]),
        atol=1e-2
    )

    np.testing.assert_allclose(
        model.bias,
        1,
        atol=1e-2
    )


def test_batch_gd_prediction():
    X_train = np.array([[1], [2], [3], [4]])
    y_train = np.array([3, 5, 7, 9])

    X_test = np.array([[5], [6]])
    expected = np.array([11, 13])

    model = LinearRegressionBatchGD(
        lr=0.1,
        max_iter=5000
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    np.testing.assert_allclose(
        predictions,
        expected,
        atol=1e-2
    )


def test_predict_before_fit():
    model = LinearRegressionBatchGD()

    with pytest.raises(ModelNotTrainedError):
        model.predict(np.array([[1]]))


def test_batch_gd_score():
    X = np.array([[1], [2], [3], [4]])
    y = np.array([3, 5, 7, 9])

    model = LinearRegressionBatchGD(
        lr=0.1,
        max_iter=5000
    )

    model.fit(X, y)

    assert model.score(X, y) > 0.999


def test_loss_history():
    X = np.array([[1], [2], [3], [4]])
    y = np.array([3, 5, 7, 9])

    model = LinearRegressionBatchGD(
        lr=0.1,
        max_iter=100
    )

    model.fit(X, y)

    assert len(model.loss_history_) == model.max_iter
