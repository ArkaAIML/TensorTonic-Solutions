import numpy as np

def _sigmoid(z):
    return np.where(
        z >= 0,
        1 / (1 + np.exp(-z)),
        np.exp(z) / (1 + np.exp(z))
    )

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    N, D = X.shape

    w = np.zeros(D)
    b = 0.0

    for _ in range(steps):

        # forward pass
        z = X @ w + b
        y_hat = _sigmoid(z)

        # ---- LOSS (this is what you missed) ----
        eps = 1e-15  # numerical stability
        loss = -np.mean(
            y * np.log(y_hat + eps) +
            (1 - y) * np.log(1 - y_hat + eps)
        )

        # gradients
        dw = (1 / N) * (X.T @ (y_hat - y))
        db = (1 / N) * np.sum(y_hat - y)

        # update
        w -= lr * dw
        b -= lr * db

    return w, float(b)