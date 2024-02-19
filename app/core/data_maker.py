""" This script is used to generate data for the model. """

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs


def make_data() -> tuple[np.ndarray, np.ndarray]:
    """Function to generate the data for the model

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The input and the output data
    """
    x_matrice, y_matrice = make_blobs(n_samples=100, centers=2, random_state=0)
    y_matrice = y_matrice.reshape((y_matrice.shape[0], 1))

    return x_matrice, y_matrice


def plot_data(x_matrice: np.ndarray, y_matrice: np.ndarray) -> None:
    """Function to plot the data

    Parameters
    ----------
    x_matrice : np.ndarray
        The input data
    y_matrice : np.ndarray
        The output data
    """
    plt.scatter(x_matrice[:, 0], x_matrice[:, 1], c=y_matrice, cmap="summer")
    plt.show()


if __name__ == "__main__":
    X, y = make_data()
    plot_data(X, y)
