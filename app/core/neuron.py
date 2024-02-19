"""Module to train an artificial neuron"""

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, log_loss

from app.core.data_maker import make_data
from app.core.functions import gradients, initialisation, model, update
from app.core.predictions import predict


def artificial_neuron(
    X: np.ndarray, y: np.ndarray, learning_rate: float = 0.1, iterations: int = 100
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Function to train an artificial neuron

    Parameters
    ----------
    X : np.ndarray
        The input data
    y : np.ndarray
        The output data
    learning_rate : float
        The learning rate
    iterations : int
        The number of iterations

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, list[float]]
        The weights and the bias and the loss
    """

    W, b = initialisation(X)

    loss = []

    for _ in tqdm(range(iterations)):
        A = model(X, W, b)
        loss.append(log_loss(y, A))
        dW, db = gradients(A, X, y)
        W, b = update(dW, db, W, b, learning_rate)

    return W, b, loss


def plot_loss(loss: list[float]) -> None:
    """Function to plot the loss

    Parameters
    ----------
    loss : list[float]
        The loss
    """
    plt.plot(loss)
    plt.show()


def perf_score(X: np.ndarray, W: np.ndarray, b: float, y: np.ndarray) -> float:
    """Function to calculate the performance score

    Parameters
    ----------
    X : np.ndarray
        The input data
    W : np.ndarray
        The weights
    b : float
        The bias
    y : np.ndarray
        The output data

    Returns
    -------
    float
        The performance score
    """
    y_pred = predict(X, W, b)
    return accuracy_score(y, y_pred)


if __name__ == "__main__":
    X, y = make_data()

    W, b, loss = artificial_neuron(X, y)
    print(W, b)
    print(perf_score(X, W, b, y))

    new_plant = np.array([[2, 1]])
    print(predict(new_plant, W, b))
