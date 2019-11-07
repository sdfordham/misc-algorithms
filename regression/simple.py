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
        if self.coefficients is None:
            raise ValueError("Need to fit data first.")
        if data.shape[1] != len(self.coefficients):
            raise ValueError("Bad data shape.")
        return np.matmul(data, self.coefficients)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    sz = 100
    e = np.random.normal(0, 10, sz)
    m = np.random.random()
    x = np.random.randint(0, sz, sz)
    y = m * x + e

    xy = np.vstack((x, y)).T
    r = Linear()
    r.fit(xy)

    plt.scatter(x, y)
    plt.plot(x, r.predict(xy[:, :-1]))
    plt.show()

