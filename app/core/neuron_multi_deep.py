"""Module to train an artificial neuron"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score, log_loss
from tqdm import tqdm

from app.core.functions_multi_deep import (
    back_propagation,
    forward_propagation,
    initialisation,
    predict,
    update,
)


def artificial_neuron_multi_deep(
    X_train, y_train, hidden_layers=[32, 32, 32], learning_rate=0.1, iterations=1000
):

    np.random.seed(0)

    dimensions = hidden_layers.copy()
    dimensions.insert(0, X_train.shape[0])
    dimensions.append(y_train.shape[0])
    parametres = initialisation(dimensions)

    train_loss = []
    train_acc = []

    for i in tqdm(range(iterations)):

        activations = forward_propagation(X_train, parametres)
        gradients = back_propagation(X_train, y_train, parametres, activations)
        parametres = update(gradients, parametres, learning_rate)

        if i % 10 == 0:
            longueur_reseau = len(parametres) // 2
            train_loss.append(
                log_loss(y_train, activations["A" + str(longueur_reseau)])
            )
            y_pred = predict(X_train, parametres)
            current_accuracy = accuracy_score(y_train.flatten(), y_pred.flatten())
            train_acc.append(current_accuracy)

    return parametres, train_loss, train_acc


def plot_loss_acc_deep(train_loss, train_acc):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 4))

    ax[0].plot(train_loss, label="train_loss")
    ax[0].legend()

    ax[1].plot(train_acc, label="train_acc")
    ax[1].legend()

    plt.show()


if __name__ == "__main__":

    X, y = make_circles(n_samples=100, noise=0.1, factor=0.3, random_state=0)
    X = X.T
    y = y.reshape((1, y.shape[0]))

    print("dimensions of X:", X.shape)
    print("dimensions of y:", y.shape)

    parameters, train_loss, train_acc = artificial_neuron_multi_deep(X, y)
    plot_loss_acc_deep(train_loss, train_acc)
