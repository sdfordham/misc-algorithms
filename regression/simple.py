import numpy as np


class Linear(object):
    def __init__(self):
        self.coefficients = None

    def fit(self, data):
        x = data[:, :-1]
        y = data[:, -1]

        # Beta = (X^t . X)^{-1}. X^t . y
        self.coefficients = np.matmul(
            np.linalg.pinv(
                np.matmul(
                    np.transpose(x),
                    x
                )
            ),
            np.matmul(
                np.transpose(x),
                y
            )
        )

    def predict(self, data):
        if data.shape[1] != len(self.coefficients):
            raise ValueError("Bad data shape.")
        if self.coefficients is None:
            raise ValueError("Need to fit to data first.")
        print(np.matmul(data, self.coefficients))


if __name__ == "__main__":
    xy = np.random.randint(0, 100, size=(3, 4))
    r = Linear()
    r.fit(xy)
    r.predict(xy[:, :-1])

