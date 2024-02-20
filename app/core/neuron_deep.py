"""Module to train an artificial neuron"""

from matplotlib import pyplot as plt
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score, log_loss
from tqdm import tqdm

from app.core.functions_deep import (
    back_propagation,
    forward_propagation,
    initialisation,
    predict,
    update,
)


def artificial_neuron_deep(X_train, y_train, n1, learning_rate=0.1, iterations=1000):

    n0 = X_train.shape[0]
    n2 = y_train.shape[0]
    parametres = initialisation(n0, n1, n2)

    train_loss = []
    train_acc = []

    for i in tqdm(range(iterations)):

        activations = forward_propagation(X_train, parametres)
        gradients = back_propagation(X_train, y_train, parametres, activations)
        parametres = update(gradients, parametres, learning_rate)

        if i % 10 == 0:
            train_loss.append(log_loss(y_train, activations["A2"]))
            y_pred = predict(X_train, parametres)
            current_accuracy = accuracy_score(y_train.flatten(), y_pred.flatten())
            train_acc.append(current_accuracy)

    return parametres, train_loss, train_acc


def plot_loss_acc_deep(train_loss, train_acc):
    plt.figure(figsize=(14, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label="Train loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label="Train accuracy")
    plt.legend()

    plt.show()


if __name__ == "__main__":

    X, y = make_circles(n_samples=100, noise=0.1, factor=0.3, random_state=0)
    X = X.T
    y = y.reshape((1, y.shape[0]))

    print("dimensions of X:", X.shape)
    print("dimensions of y:", y.shape)

    plt.scatter(X[0, :], X[1, :], c=y, cmap="summer")
    plt.show()

    parameters, train_loss, train_acc = artificial_neuron_deep(
        X, y, n1=2, learning_rate=0.1, iterations=1000
    )
    plot_loss_acc_deep(train_loss, train_acc)
