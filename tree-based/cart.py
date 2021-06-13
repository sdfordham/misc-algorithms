import operator
from collections import defaultdict
import numpy as np
from sklearn.tree import DecisionTreeClassifier


class Region:
    def __init__(self, data: np.array, column_idx: int, split_val: float, oper: operator):
        self.conditions = [(column_idx, split_val, oper)]
        self.data = data
        self.data_restricted = self._split_data(data)
        self.gini_idx = self._gini_idx
        self.point_count = self._point_count

    def add_region(self, region):
        for (idx, val, oper) in region.conditions:
            if (idx, val, oper) not in self.conditions:
                self.conditions.append((idx, val, oper))

        # Update
        self.data_restricted = self._split_data(self.data)
        self.gini_idx = self._gini_idx
        self.point_count = self._point_count

    @property
    def _gini_idx(self) -> float:
        _sum = 0
        for t in np.unique(self.data_restricted[:, -1]):
            p_mk = self._perc_match(t)
            _sum = _sum + p_mk * (1 - p_mk)
        return _sum

    @property
    def best_label(self) -> float:
        max_p, klass = 0, None
        for t in np.unique(self.data_restricted[:, -1]):
            p = self._perc_match(t)
            if p > max_p:
                max_p, klass = p, t
        return klass

    @property
    def _point_count(self) -> int:
        return self.data_restricted.shape[0]

    def _perc_match(self, label: float) -> float:
        matched = self.data_restricted[self.data_restricted[:, -1] == label].shape[0]
        return matched / self.data_restricted.shape[0]
    
    def _split_data(self, arr: np.array) -> np.array:
        _arr = arr.copy()
        for (idx, val, oper) in self.conditions:
            _arr = _arr[oper(_arr[:, idx], val)]
        return _arr


def cost_complexity(regions: list[Region]) -> float:
    return sum([r.gini_idx * r.point_count for r in regions])


def compare_sklearn(max_depth=1):
    arr = np.loadtxt(r"tree-based\spam.data")
    
    X, y = arr[:, :-1], arr[:, -1]
    clf = DecisionTreeClassifier(max_depth=max_depth)
    clf.fit(X, y)

    print("sklearn splits:")
    for col, threshold in zip(clf.tree_.feature, clf.tree_.threshold):
        if col >= 0:
            print(f"\tX[:, {col}] <= {threshold:.5f}")

def main():
    arr = np.loadtxt(r"tree-based\spam.data")

    costs = defaultdict(list)
    cols = list(range(arr.shape[1] - 1))
    min_cost, best_col, best_split = 1e6, None, None
    for col in cols:
        for split in np.unique(arr[:, col]):
            left = Region(arr, col, split, operator.le)
            right = Region(arr, col, split, operator.gt)
            cc = cost_complexity([left, right])
            if cc < min_cost:
                min_cost, best_col, best_split = cc, col, split
    

    print(f"Best first split: X[:, {best_col}] <= {best_split} (cost complexity: {min_cost:.5f}).")

    best_left = Region(arr, best_col, best_split, operator.le)
    best_right = Region(arr, best_col, best_split, operator.gt)
    cols.pop(best_col)

    min_cost, best_col, best_split = 1e6, None, None
    for col in cols:
        for split in np.unique(arr[:, col]):
            left = Region(arr, col, split, operator.le)
            left.add_region(best_left)
            right = Region(arr, col, split, operator.gt)
            right.add_region(best_left)
            cc = cost_complexity([left, right])
            if cc < min_cost:
                min_cost, best_col, best_split = cc, col, split

    print(f"Next split on left: X[:, {best_col}] <= {best_split} (cost complexity: {min_cost:.5f}).")

    min_cost, best_col, best_split = 1e6, None, None
    for col in cols:
        for split in np.unique(arr[:, col]):
            left = Region(arr, col, split, operator.le)
            left.add_region(best_right)
            right = Region(arr, col, split, operator.gt)
            right.add_region(best_right)
            cc = cost_complexity([left, right])
            if cc < min_cost:
                min_cost, best_col, best_split = cc, col, split

    print(f"Next split on left: X[:, {best_col}] <= {best_split} (cost complexity: {min_cost:.5f}).")

    compare_sklearn(2)


if __name__ == "__main__":
    main()