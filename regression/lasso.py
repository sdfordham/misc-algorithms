import numpy as np
from simple import Linear


class LassoRegression(Linear):
    def __init__(self, **kwargs):
        super().__init__()
        self.alpha = kwargs.get('alpha', 1.0)
        self.threshold = kwargs.get('threshold', 0.1)
        self.iterations = kwargs.get('iterations', 10)

    def fit(self, data):
        x = data[:, :-1]
        y = data[:, -1]
        self.coefficients = np.zeros(x.shape[1])

        it = 0
        delta_rss = self.threshold + 1
        while it < self.iterations and delta_rss > self.threshold:
            old_rss = self._rss(x, y)
            self._update_coefficients(x, y)
            new_rss = self._rss(x, y)
            delta_rss = abs(old_rss - new_rss)
            it = it + 1

    def _residual(self, x, y):
        return y - np.matmul(x, self.coefficients)

    def _rss(self, x, y):
        return np.dot(self._residual(x, y), self._residual(x, y))

    def _update_coefficients(self, x, y):
        a = np.array([np.dot(x[:, i], self._residual(x, y)) for i in
            range(x.shape[1])])
        N = x.shape[0]
        op = np.vectorize(self._soft_threshold_op)
        self.coefficients = op(self.coefficients + (1 / N) * a, self.alpha)

    @staticmethod
    def _soft_threshold_op(z, g):
        if z > 0 and g < abs(z):
            return z - g
        elif z < 0 and g < abs(z):
            return z + g
        else:
            return 0


if __name__ == "__main__":
    xy = np.random.randint(0, 100, size=(3, 6))
    r = LassoRegression(alpha=0.75)
    r.fit(xy)
    r.predict(xy[:, :-1])
