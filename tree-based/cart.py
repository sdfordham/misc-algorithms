import operator
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


class Node:
    def __init__(self, root=False, parent=None) -> None:
        self.parent = parent
        self.regions = list()
        self.left_child = None
        self.right_child = None


class Tree:
    def __init__(self, root: Node) -> None:
        self.root = root


def get_best_split(arr: np.array, cols: list, node: Node):
    min_cost, best_col, best_split = 1e6, None, None
    best_left, best_right = None, None
    for col in cols:
        for split in np.unique(arr[:, col]):
            left_region = Region(arr, col, split, operator.le)
            for region in node.regions:
                left_region.add_region(region)
            right_region = Region(arr, col, split, operator.gt)
            for region in node.regions:
                right_region.add_region(region)
            cc = cost_complexity([left_region, right_region])
            if cc < min_cost:
                min_cost, best_col, best_split = cc, col, split
                best_left, best_right = left_region, right_region
    return (best_left, best_right), best_col, best_split


def split_node(node: Node, left_region: Region, right_region: Region) -> tuple:
    left_child, right_child = Node(parent=node), Node(parent=node)
    node.left_child, node.right_child = left_child, right_child
    left_child.regions.append(left_region)
    right_child.regions.append(right_region)
    return left_child, right_child

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

    root = Node()
    tree = Tree(root=root)
    cols = list(range(arr.shape[1] - 1))

    (best_left, best_right), best_col, best_split = get_best_split(arr, cols, root)
    left_child, right_child = split_node(root, best_left, best_right)
    print(f"Best first split: X[:, {best_col}] <= {best_split}.")

    cols.pop(best_col)

    (best_left, best_right), best_col, best_split = get_best_split(arr, cols, left_child)
    left_left_child, right_left_child = split_node(left_child, best_left, best_right)
    print(f"Next split on left: X[:, {best_col}] <= {best_split}")

    (best_left, best_right), best_col, best_split = get_best_split(arr, cols, right_child)
    left_right_child, right_right_child = split_node(right_child, best_left, best_right)
    print(f"Next split on right: X[:, {best_col}] <= {best_split}")

    compare_sklearn(2)


if __name__ == "__main__":
    main()