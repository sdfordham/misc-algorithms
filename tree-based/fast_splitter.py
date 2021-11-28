import numpy as np
from collections import namedtuple


Splitter = namedtuple('Splitter', ['split_value', 'column_index', 'cost_complexity'])


def gini_idx(n_0: int, n_tot: int) -> float:
    p_0 = n_0 / n_tot
    return 2 * p_0 * (1 - p_0)


def get_best_splitter(arr: np.ndarray) -> Splitter:
    n_tot = arr.shape[0]
    n_tot_0 = sum(arr[:, -1] == 0.)
    n_tot_1 = sum(arr[:, -1] == 1.)

    best_splitters = list()
    for col in range(arr.shape[1] - 1):
        arr_sorted = arr[arr[:, col].argsort()]  # sort
        X, y = arr_sorted[:, col], arr_sorted[:, -1]

        n_0l, n_1l = 0, 0
        n_0r, n_1r = n_tot_0, n_tot_1

        min_split, min_cc = None, 1e6
        for i in range(n_tot):
            XX, yy = X[i], y[i]

            if yy == 1:
                n_1l += 1
                n_1r -= 1
            elif yy == 0:
                n_0l += 1
                n_0r -= 1

            if i == n_tot - 1 or XX != X[i + 1]:
                cc_left = (n_0l + n_1l) * gini_idx(n_0l, n_0l + n_1l)
                cc_right = (n_0r + n_1r) * gini_idx(n_0r, n_0r + n_1r)
                if cc_left + cc_right < min_cc:
                    min_cc = cc_left + cc_right
                    min_split = XX
        best_splitters.append(
            Splitter(min_split, col, min_cc)
        )
    return sorted(best_splitters, key=lambda x: x.cost_complexity)[0]


if __name__ == '__main__':
    arr = np.loadtxt('spam.data')
    res = get_best_splitter(arr)
    print(res)
