from time import time
from dataclasses import dataclass
from math import sqrt
from typing import Literal, Union


Pm1 = Union[Literal[-1], Literal[1]]


@dataclass
class QuadInt:
    a: float
    b: float
    D: float

    def pow(self, n: int) -> 'QuadInt':
        a_, b_ = self.a, self.b
        res = QuadInt(a=self.a, b=self.b, D=self.D)
        for _ in range(n - 1):
            old_a = res.a
            res.a = old_a * a_ + res.b * b_ * res.D
            res.b = old_a * b_ + a_ * res.b
        return res

    def divide(self, x: Union[float, int]) -> 'QuadInt':
        return QuadInt(
            a = self.a / x,
            b = self.b / x,
            D = self.D
        )

    @property
    def pell(self) -> Union[int, float]:
        return self.a * self.a - self.D * self.b * self.b


def calc_unit_pow(x: int, y: int, sigma: Pm1) -> int:
    if x % 2 == 0:
        if sigma == 1:
            return 1 if y % 2 == 0 else 2
        return 2
    else:
        return 3 if sigma == 1 else 6


def fundamental_unit(D: int) -> tuple[QuadInt, int, Pm1]:
    """
    Fundamental unit for the group of units of Q(sqrt(D))
    """
    # PPP := P_(i+1), PP := P_i, P := P_(i-1) etc.
    def floor_delta(n1: float, n2: float, d: int) -> int:
        return int( (n1 + n2) / d)
    s = 2 if D % 4 == 0 or D % 4 == 1 else 1
    q = 1 if D % 4 == 1 else 0

    PP, QQ = q, s
    qq = floor_delta(int(sqrt(D)), q, s)
    B, BB = 0, 1
    G, GG = s, s * qq - q

    p, QQQ = 0, -1
    while(QQQ != s):
        PPP = qq * QQ - PP
        QQQ = floor_delta(D, - PPP * PPP, QQ) 
        qqq = floor_delta(PPP, int(sqrt(D)), QQQ)
        GGG, BBB = qqq * GG + G, qqq * BB + B
        PP, QQ, qq = PPP, QQQ, qqq
        GG, G, BB, B = GGG, GG, BBB, BB
        p += 1
    sgn: Pm1 = 1 if p % 2 == 0 else -1
    return QuadInt(G, B, D), s, sgn
