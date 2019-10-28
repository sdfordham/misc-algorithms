import numpy as np
from simple import Linear


class RidgeRegression(Linear):
    def __init__(self, **kwargs):
        super().__init__()
        self.alpha = kwargs.get('alpha', 1.0)

    def fit(self, data):
        x = data[:, :-1]
        y = data[:, -1]
        unit = np.identity(x.shape[1])

        # Beta = (X^t . X + alpha . I)^{-1} . X^t . y
        self.coefficients = np.matmul(
            np.linalg.pinv(
                np.add(
                    np.matmul(
                        np.transpose(x),
                        x
                    ),
                    self.alpha * unit
                )
            ),
            np.matmul(
                np.transpose(x),
                y
            )
        )


if __name__ == "__main__":
    xy = np.random.randint(0, 100, size=(3, 4))
    r = RidgeRegression(alpha=0.75)
    r.fit(xy)
    r.predict(xy[:, :-1])




