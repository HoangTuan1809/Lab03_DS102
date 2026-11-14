import numpy as np

class LinearRegression:
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64).reshape(-1,1)
        n_samples = X.shape[0]

        X_b = np.c_[np.ones((n_samples,1)), X]
        X_T_X = X_b.T.dot(X_b)
        X_T_y = X_b.T.dot(y)

        theta = np.linalg.pinv(X_T_X).dot(X_T_y)

        self.bias = theta[0,0]
        self.weights = theta[1:]

    def predict(self, X):
        X = np.array(X, dtype=np.float64)
        return (X @ self.weights + self.bias).flatten()

    def evaluate(self, X, y):
        y_true = np.array(y, dtype=np.float64)
        y_pred = self.predict(X)

        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))

        return{
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
        }