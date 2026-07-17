import numpy as np
import pytest

from linear_regression.stochastic_gd import LinearRegressionSGD
from linear_regression.exceptions import ModelNotTrainedError


def test_sgd_initialization():
    model = LinearRegressionSGD()

    assert model.weights is None
    assert model.bias is None
    assert model.learning_rate == 0.01
    assert model.max_iter == 10000


def test_sgd_fit():
    """
    Check model learns parameters after training
    """

    # y = 2x + 1
    X = np.array([
        [1],
        [2],
        [3],
        [4]
    ])

    y = np.array([3, 5, 7, 9])


    model = LinearRegressionSGD(
        lr=0.001,
        max_iter=200,
        random_state=42
    )

    model.fit(X, y)


    assert model.weights is not None
    assert model.bias is not None



def test_sgd_learns_linear_relationship():
    """
    Check if SGD learns correct relationship
    """

    X = np.array([
        [1],
        [2],
        [3],
        [4]
    ])

    y = np.array([3, 5, 7, 9])


    model = LinearRegressionSGD(
        lr=0.01,
        max_iter=500,
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



def test_sgd_prediction():

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


    model = LinearRegressionSGD(
        lr=0.01,
        max_iter=500,
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

    model = LinearRegressionSGD()

    X = np.array([[1]])


    with pytest.raises(ModelNotTrainedError):
        model.predict(X)



def test_sgd_score():

    X = np.array([
        [1],
        [2],
        [3],
        [4]
    ])

    y = np.array([3, 5, 7, 9])


    model = LinearRegressionSGD(
        lr=0.01,
        max_iter=500,
        random_state=42
    )


    model.fit(X, y)


    score = model.score(X, y)


    assert score > 0.99



def test_loss_history():

    X = np.array([
        [1],
        [2],
        [3]
    ])

    y = np.array([3, 5, 7])


    iterations = 50


    model = LinearRegressionSGD(
        lr=0.01,
        max_iter=iterations,
        random_state=42
    )


    model.fit(X, y)


    assert len(model.loss_history_) == iterations
