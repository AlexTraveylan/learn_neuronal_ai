""" This module contains the functions used in the main module """

import numpy as np


def initialisation(x_matrice: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Function to initialize the weights and the bias

    Parameters
    ----------
    x_matrice : np.ndarray
        The input data

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The weights and the bias
    """
    weights = np.random.randn(x_matrice.shape[1], 1)
    bias = np.random.randn(1)
    return weights, bias


def model(x_matrice: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Function to compute the model

    Parameters
    ----------
    x_matrice : np.ndarray
        The input data
    weights : np.ndarray
        The weights
    bias : np.ndarray
        The bias

    Returns
    -------
    np.ndarray
        The activation matrice
    """
    z_vector = x_matrice.dot(weights) + bias
    activation_matrice = 1 / (1 + np.exp(-z_vector))

    return activation_matrice


def gradients(
    A: np.ndarray, X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Function to compute the gradients

    Parameters
    ----------
    A : np.ndarray
        The activation matrice
    X : np.ndarray
        The input data
    y : np.ndarray
        The output data

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The gradients
    """

    dW = (1 / y.shape[0]) * np.dot(X.T, (A - y))
    db = (1 / y.shape[0]) * np.sum(A - y)

    return dW, db


def update(
    dW: np.ndarray, db: np.ndarray, W: np.ndarray, b: np.ndarray, learning_rate: float
) -> tuple[np.ndarray, np.ndarray]:
    """Function to update the weights and the bias

    Parameters
    ----------
    dW : np.ndarray
        The gradients of the weights
    db : np.ndarray
        The gradients of the bias
    W : np.ndarray
        The weights
    b : np.ndarray
        The bias
    learning_rate : float
        The learning rate

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The updated weights and bias
    """
    W = W - learning_rate * dW
    b = b - learning_rate * db

    return W, b
