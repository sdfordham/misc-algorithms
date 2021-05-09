import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt

data = np.loadtxt(r"spam.data")

def perc_match(arr, label):
    """
    Proportion of label `label` observations in `arr`
    """
    return arr[arr[:, -1] == label].shape[0] / arr.shape[0]

def node_class(arr):
    """
    Find class that maximizes argmax_k perc_match
    """
    max_p, klass = 0, None
    for t in np.unique(arr[:, -1]):
        p = perc_match(arr, t)
        if p > max_p:
            max_p, klass = p, t
    return klass

def gini_idx(arr):
    """
    Gini index
    """
    _sum = 0
    for t in np.unique(arr[:, -1]):
        p_mk = perc_match(arr, t)
        _sum = _sum + p_mk * (1 - p_mk)
    return _sum


class Region:
    def __init__(self, column_idx: int, split_val: float, data):
        self.conditions = [(column_idx, split_val)]

    def add_region(self, region):
        for (idx, val) in region.conditions:
            if (idx, val) not in self.conditions:
                self.conditions.append((idx, val))

    @property
    def gini_idx(data):
        pass

def main():
    COL = 0
    node_values_l = list()
    node_values_r = list()
    for s in np.unique(data[:, COL]):
        left = data[data[:, COL] <= s]
        right = data[data[:, COL] > s]

        left_class = node_class(left)
        right_class = node_class(right)

        node_values_l.append(gini_idx(left))
        node_values_r.append(gini_idx(right))

    plt.plot(node_values_l)
    plt.plot(node_values_r)

    plt.show()


if __name__ == "__main__":
    main()
