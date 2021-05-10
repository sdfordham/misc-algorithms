import operator
import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt


class Region:
    def __init__(self, column_idx: int, split_val: float, oper: operator):
        self.conditions = [(column_idx, split_val, oper)]

    def add_region(self, region):
        for (idx, val, oper) in region.conditions:
            if (idx, val, oper) not in self.conditions:
                self.conditions.append((idx, val, oper))

    def gini_idx(self, arr: np.array) -> float:
        _sum = 0
        _arr = self._split_data(arr)
        for t in np.unique(_arr[:, -1]):
            p_mk = self._perc_match(_arr, t)
            _sum = _sum + p_mk * (1 - p_mk)
        return _sum

    def best_label(self, arr: np.array) -> float:
        max_p, klass = 0, None
        _arr = self._split_data(arr)
        for t in np.unique(_arr[:, -1]):
            p = self._perc_match(_arr, t)
            if p > max_p:
                max_p, klass = p, t
        return klass

    def _split_data(self, arr: np.array) -> np.array:
        _arr = arr.copy()
        for (idx, val, oper) in self.conditions:
            _arr = _arr[oper(_arr[:, idx], val)]
        return _arr

    def _perc_match(self, arr: np.array, label: float) -> float:
        return arr[arr[:, -1] == label].shape[0] / arr.shape[0]


def main():
    data = np.loadtxt(r"tree-based\spam.data")
    COL = 0
    node_values_l = list()
    node_values_r = list()
    for s in np.unique(data[:, COL]):
        left = Region(COL, s, operator.le)
        right = Region(COL, s, operator.gt)

        left_class = left.best_label(data)
        right_class = right.best_label(data)

        node_values_l.append(left.gini_idx(data))
        node_values_r.append(right.gini_idx(data))

    plt.plot(node_values_l)
    plt.plot(node_values_r)

    plt.show()


if __name__ == "__main__":
    main()
