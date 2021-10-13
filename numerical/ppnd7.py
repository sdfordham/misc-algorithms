from math import sqrt, log


SPLIT1 = 0.425e0
SPLIT2 = 5.0e0

CONST1 = 0.180625e0
CONST2 = 1.6e0

AO = 3.3871327179e0
A1 = 5.0434271938e1
A2 = 1.5929113202e2
A3 = 5.9109374720e1
B1 = 1.7895169469e1
B2 = 7.8757757664e1
B3 = 6.7187563600e1

C0 = 1.4234372777e0
C1 = 2.7568153900e0
C2 = 1.3067284816e0
C3 = 1.7023821103e-1

D1 = 7.3700164250e-1
D2 = 1.2021132975e-1

E0 = 6.6579051150e0
E1 = 3.0812263860e0
E2 = 4.2868294337e-1
E3 = 1.7337203997e-2
F1 = 2.4197894225e-1
F2 = 1.2258202635e-2


def PPND7(p: float) -> float:
    """
    Wichura's Percentage Points of the Normal
        Distribution algorithm (PPND7)
    """
    q = p - 0.5

    if abs(q) <= SPLIT1:
        r = CONST1 - q * q
        ppnd7_num = q * (((A3 * r + A2) * r + A1) * r + AO)
        ppnd7_denom = ((B3 * r + B2) * r + B1) * r + 1.0
        return ppnd7_num / ppnd7_denom
    else:
        r = p if q < 0.0 else 1.0 - p
        if r <= 0.0:
            raise ValueError("Must have 0 < p < 1")
        r = sqrt(-log(r))
        if r <= SPLIT2:
            r = r - CONST2
            ppnd7_num = (((C3 * r + C2) * r + C1) * r + C0)
            ppnd7_denom = ((D2 * r + D1) * r + 1.0)
        else:
            r = r - SPLIT2
            ppnd7_num = (((E3 * r + E2) * r + E1) * r + E0)
            ppnd7_denom = ((F2 * r + F1) * r + 1.0)
        sign = 1.0 if q >= 0.0 else -1.0
        return sign * ppnd7_num / ppnd7_denom


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    step = 1e-3
    grid = np.arange(step, 1, step)

    plt.figure(figsize=(15, 10))
    plt.plot(grid, [PPND7(x) for x in grid])
    plt.plot(grid, [norm.ppf(x) for x in grid])
