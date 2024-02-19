""" Module for making predictions"""

import numpy as np

from app.core.functions import model


def predict(X: np.ndarray, W: np.ndarray, b: float) -> bool:
    """Function to make predictions

    Parameters
    ----------
    X : np.ndarray
        The input data
    W : np.ndarray
        The weights
    b : float
        The bias

    Returns
    -------
    bool
        The prediction
    """
    A = model(X, W, b)
    return A >= 0.5
