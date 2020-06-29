import numpy as np


def normalise(X):
    offset = np.average(X, axis=0)
    scale = np.std(X, axis=0)
    X_normal = (X - offset) / scale
    return X_normal, offset, scale
