import numpy as np
from simple import Linear


class LassoRegression(Linear):
    def __init__(self, **kwargs):
        super().__init__()
        self.alpha = kwargs.get('alpha', 1.0)
        self.threshold = kwargs.get('threshold', 0.1)
        self.iterations = kwargs.get('iterations', 1000)

    def fit(self, data):
        x = data[:, :-1]
        y = data[:, -1]
        self.coefficients = np.zeros(x.shape[0])

        it = 0
        delta_rss = self.threshold + 1
        while it < self.iterations and delta_rss > self.threshold:
            old_rss = self._rss(x, y)
            self._update_coefficients(x, y)
            new_rss = self._rss(x, y)

            delta_rss = abs(old_rss - new_rss)

            it = it + 1

    def _rss(self, x, y):
        delta = y - np.matmul(x, self.coefficients)
        return np.matmul(np.transpose(delta), delta)

    def _update_coefficients(self, x, y):
        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2929880/
        pass

    @staticmethod
    def _soft_threshold_op(z, g):
        if z > 0 and g < abs(z):
            return z - g
        elif z < 0 and g < abs(z):
            return z + g
        else:
            return 0

