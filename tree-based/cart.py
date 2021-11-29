from time import time
from typing import Optional
from collections import namedtuple
import numpy as np
from sklearn.tree import DecisionTreeClassifier


Splitter = namedtuple('Splitter', ['split_value', 'column_index', 'cost_complexity'])


def timing(f: callable):
    def wrapped(*args, **kwargs):
        start = time()
        res = f(*args, **kwargs)
        end = time()
        print(f'{f.__name__} took {end - start:.4f}')
        return res
    return wrapped


def gini_idx(n_0: int, n_tot: int) -> float:
    p_0 = n_0 / n_tot
    return 2 * p_0 * (1 - p_0)


@timing
def get_best_splitter(arr: np.ndarray,
                      ignore_cols: Optional[list[int]] = None) -> Splitter:
    n_tot = arr.shape[0]
    n_tot_0 = sum(arr[:, -1] == 0.)
    n_tot_1 = sum(arr[:, -1] == 1.)

    best_splitters = list()
    for col in range(arr.shape[1] - 1):
        if ignore_cols and col in ignore_cols:
            continue

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


@timing
def compare_sklearn(arr: np.ndarray, max_depth=1) -> None:
    X, y = arr[:, :-1], arr[:, -1]
    clf = DecisionTreeClassifier(max_depth=max_depth)
    clf.fit(X, y)

    print("sklearn splits:")
    for col, threshold in zip(clf.tree_.feature, clf.tree_.threshold):
        if col >= 0:
            print(f"\tX[:, {col}] <= {threshold:.5f}")


def main():
    arr = np.loadtxt('spam.data')
    ignore_cols = list()

    res_root = get_best_splitter(arr, ignore_cols=ignore_cols)
    print(res_root)
    ignore_cols.append(res_root.column_index)

    arr_left = arr[arr[:, res_root.column_index] <= res_root.split_value]
    res_left = get_best_splitter(arr_left, ignore_cols=ignore_cols)
    print(res_left)

    arr_right = arr[arr[:, res_root.column_index] > res_root.split_value]
    res_right = get_best_splitter(arr_right, ignore_cols=ignore_cols)
    print(res_right)

    compare_sklearn(arr, 2)


if __name__ == "__main__":
    main()

