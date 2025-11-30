# MIT License © 2025 ISD NightNova

import numpy as np

class AI:
    """Core machine learning mathematics and utility functions."""

    # --------------------- LOSS FUNCTIONS ---------------------

    def mean_squared_error(self, y_true, y_pred):
        """Mean Squared Error."""
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        return np.mean((y_true - y_pred) ** 2)

    def cross_entropy_loss(self, y_true, y_pred):
        """Binary cross entropy loss."""
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    # -------------------- ACTIVATION FUNCTIONS --------------------

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        x = np.asarray(x)
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    # ---------------------- NEURAL NETWORK BASICS ----------------------

    def forward_propagation(self, X, W, b):
        return np.dot(X, W) + b

    def backward_propagation(self, X, y, W, b, lr=0.01):
        """Simple logistic regression backprop."""
        m = X.shape[0]
        A = self.sigmoid(self.forward_propagation(X, W, b))
        dW = (1 / m) * np.dot(X.T, (A - y))
        db = np.mean(A - y)
        W -= lr * dW
        b -= lr * db
        return W, b

    # ---------------------- REGRESSION MODELS ----------------------

    def linear_regression(self, X, y):
        """Least squares linear regression."""
        X = np.c_[np.ones(X.shape[0]), X]
        return np.linalg.lstsq(X, y, rcond=None)[0]

    def polynomial_regression(self, X, y, degree=2):
        """Polynomial regression using Vandermonde matrix."""
        X_poly = np.vander(X, degree + 1)
        return np.linalg.lstsq(X_poly, y, rcond=None)[0]

    # ---------------------- K-MEANS CLUSTERING ----------------------

    def k_means(self, X, k, max_iters=100):
        """Basic K-Means clustering."""
        X = np.asarray(X)
        centroids = X[np.random.choice(X.shape[0], k, replace=False)]

        for _ in range(max_iters):
            distances = np.linalg.norm(X[:, None] - centroids, axis=2)
            labels = np.argmin(distances, axis=1)

            new_centroids = np.array([
                X[labels == j].mean(axis=0) if np.any(labels == j) else centroids[j]
                for j in range(k)
            ])

            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids

        return centroids, labels
