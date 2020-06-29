from abc import abstractmethod, ABCMeta
import numpy as np


class BaseRegression(metaclass=ABCMeta):
    def __init__(self):
        self.coefficients = None

    @abstractmethod
    def fit(self, X, y):
        pass

    def predict(self, X):
        if self.coefficients is None:
            raise ValueError("Need to fit data first.")
        if X.shape[1] != len(self.coefficients):
            raise ValueError("Bad data shape.")
        return np.matmul(X, self.coefficients)


class LinearRegression(BaseRegression):
    def fit(self, X, y):
        # Beta = (X^t . X)^{-1}. X^t . y
        self.coefficients = np.matmul(
            np.linalg.pinv(
                np.matmul(
                    np.transpose(X),
                    X
                )
            ),
            np.matmul(
                np.transpose(X),
                y
            )
        )


class RidgeRegression(BaseRegression):
    def __init__(self, **kwargs):
        super().__init__()
        self.alpha = kwargs.get('alpha', 1.0)

    def fit(self, X, y):
        unit = np.identity(X.shape[1])

        # Beta = (X^t . X + alpha . I)^{-1} . X^t . y
        self.coefficients = np.matmul(
            np.linalg.pinv(
                np.add(
                    np.matmul(
                        np.transpose(X),
                        X
                    ),
                    self.alpha * unit
                )
            ),
            np.matmul(
                np.transpose(X),
                y
            )
        )
