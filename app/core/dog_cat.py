from app.adapters.utilities import load_data
from app.core.neuron import artificial_neuron, perf_score, plot_loss


def normalise(X):
    return X / 255


def flatten(X):
    return X.reshape(X.shape[0], -1)


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()
    print(X_train.shape, y_train.shape)

    X_train = normalise(flatten(X_train))
    X_test = normalise(flatten(X_test))

    W, b, loss = artificial_neuron(X_train, y_train, learning_rate=0.01, iterations=1000)
    plot_loss(loss)

    print(W, b)
    print(perf_score(X_test, W, b, y_test))
