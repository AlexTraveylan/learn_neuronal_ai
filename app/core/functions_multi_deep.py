""" This module contains the functions used in the main module """

import numpy as np


def initialisation(dimensions):

    parameters = {}

    for couche in range(1, len(dimensions)):
        W = np.random.randn(dimensions[couche], dimensions[couche - 1])
        b = np.zeros((dimensions[couche], 1))
        parameters["W" + str(couche)] = W
        parameters["b" + str(couche)] = b

    return parameters


def forward_propagation(X, parameters):

    activations = {"A0": X}

    longueur_reseau = len(parameters) // 2

    for couche in range(1, longueur_reseau + 1):
        W = parameters["W" + str(couche)]
        b = parameters["b" + str(couche)]
        Z = W.dot(activations["A" + str(couche - 1)]) + parameters["b" + str(couche)]
        A = 1 / (1 + np.exp(-Z))
        activations["A" + str(couche)] = A

    return activations


def back_propagation(X, y, parameters, activations):

    m = y.shape[1]
    longueur_reseau = len(parameters) // 2

    dZ = activations["A" + str(longueur_reseau)] - y

    gradients = {}

    for couche in reversed(range(1, longueur_reseau + 1)):
        A = activations["A" + str(couche - 1)]
        dW = 1 / m * dZ.dot(A.T)
        db = 1 / m * np.sum(dZ, axis=1, keepdims=True)

        if couche > 1:
            # dZ0 n'a pas de sens
            dZ = parameters["W" + str(couche)].T.dot(dZ) * (A * (1 - A))

        gradients["dW" + str(couche)] = dW
        gradients["db" + str(couche)] = db

    return gradients


def update(gradients, parameters, learning_rate):

    longueur_reseau = len(parameters) // 2

    for couche in range(1, longueur_reseau + 1):
        parameters["W" + str(couche)] -= learning_rate * gradients["dW" + str(couche)]
        parameters["b" + str(couche)] -= learning_rate * gradients["db" + str(couche)]

    return parameters


def predict(X, parameters):
    longueur_reseau = len(parameters) // 2

    activations = forward_propagation(X, parameters)
    return activations["A" + str(longueur_reseau)] > 0.5
