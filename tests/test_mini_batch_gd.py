import numpy as np
import pytest

from linear_regression.mini_batch_gd import LinearRegressionMiniBatchGD
from linear_regression.exceptions import ModelNotTrainedError


def test_minibatch_gd_initialization():
    model = LinearRegressionMiniBatchGD()

    assert model.weights is None
    assert model.bias is None
    assert model.learning_rate == 0.01
    assert model.max_iter == 10000
    assert model.batch_size == 32


def test_invalid_batch_size():
    with pytest.raises(ValueError):
        LinearRegressionMiniBatchGD(batch_size=0)


def test_minibatch_gd_fit():
    X = np.array([
        [1],
        [2],
        [3],
        [4]
    ])

    y = np.array([3, 5, 7, 9])

    model = LinearRegressionMiniBatchGD(
        lr=0.01,
        max_iter=500,
        batch_size=2,
        random_state=42
    )

    model.fit(X, y)

    assert model.weights is not None
    assert model.bias is not None


def test_minibatch_gd_learns_linear_relationship():
    X = np.array([
        [1],
        [2],
        [3],
        [4]
    ])

    y = np.array([3, 5, 7, 9])

    model = LinearRegressionMiniBatchGD(
        lr=0.01,
        max_iter=500,
        batch_size=2,
        random_state=42
    )

    model.fit(X, y)

    np.testing.assert_allclose(
        model.weights,
        np.array([2]),
        atol=0.1
    )

    np.testing.assert_allclose(
        model.bias,
        1,
        atol=0.1
    )


def test_minibatch_prediction():
    X_train = np.array([
        [1],
        [2],
        [3],
        [4]
    ])

    y_train = np.array([3, 5, 7, 9])

    X_test = np.array([
        [5],
        [6]
    ])

    expected = np.array([11, 13])

    model = LinearRegressionMiniBatchGD(
        lr=0.01,
        max_iter=500,
        batch_size=2,
        random_state=42
    )

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    np.testing.assert_allclose(
        prediction,
        expected,
        atol=0.2
    )


def test_predict_before_fit():
    model = LinearRegressionMiniBatchGD()

    X = np.array([[1]])

    with pytest.raises(ModelNotTrainedError):
        model.predict(X)


def test_minibatch_score():
    X = np.array([
        [1],
        [2],
        [3],
        [4]
    ])

    y = np.array([3, 5, 7, 9])

    model = LinearRegressionMiniBatchGD(
        lr=0.01,
        max_iter=500,
        batch_size=2,
        random_state=42
    )

    model.fit(X, y)

    assert model.score(X, y) > 0.99


def test_loss_history():
    X = np.array([
        [1],
        [2],
        [3]
    ])

    y = np.array([3, 5, 7])

    iterations = 50

    model = LinearRegressionMiniBatchGD(
        lr=0.01,
        max_iter=iterations,
        batch_size=2,
        random_state=42
    )

    model.fit(X, y)

    assert len(model.loss_history_) == iterations
