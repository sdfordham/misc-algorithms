import pandas as pd
from ..regression.ancillary import normalise

def lars(X: pd.DataFrame, y: pd.DataFrame, s: float):
    """
    LARS algorithm ref. Lasso book
    """
    r = [y.copy()]

    beta_0 = [0] * X.shape[1]
    beta = [beta_0]

    Xb = normalise(X.copy())

    # A = index i such that (Xb^T)_i r_0 >= (Xb^T)_j r_0
    # for all j = 1, ... , X.shape[1]

    # alpha_0 = (Xb^T)_i r_0    <- determine idx

    # Xb_A matrix [Xb_j for j in A]    <- fn...

    # for i in range(1, min(X.shape[0]-1, X.shape[1]))
    #       delta = ( 1 / alpha_(i-1) ) * (X_A^T X_A)^(-1) * (X_A^T) * r_(i-1)
    #       Delta = vector( delta_j for j in A,
    #                       0       for j not in A)
    #       r(lambda) := r_(i-1) - (lambda_i - lambda) * X * Delta
    #       alpha^* = alpha_(i-1) / S
    #       (STEP ALONG THIS PATH LENGTH S)
    #       j = 0
    #       while X_k^T r(j * alpha^*) <= j alpha^* (all k in A)
    #           j = j + 1
    #       A <- A union {index k such that Xb^T r(j * alpha^*) > j * alpha^*}
    #       alpha_i = j * alpha^*
    #       beta_i = beta_(i-1) + (alpha_(i-1) - alpha_i) * Delta
    #       r_i = y - X * beta
    # return (all the alpha_i and the beta_i)   <- sklearn


