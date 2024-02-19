"""Main module for the application."""
from sklearn.metrics import log_loss
from app.core.data_maker import make_data
from app.core.functions import gradients, initialisation, model

if __name__ == "__main__":
    X, y = make_data()
    W, b = initialisation(X)
    A = model(X, W, b)

    print(log_loss(y, A))

    dW, db = gradients(A, X, y)
    print(dW.shape, db.shape)
