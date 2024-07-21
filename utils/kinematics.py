import numpy as np


def kallen(a, b, c):
    return a**2 + b**2 + c**2 - 2 * (a * b + a * c + b * c)


def get_s(m1, m2, E1, E2, costh):
    # costh = cos(theta)
    p = lambda E, m: (E**2 - m**2) ** 0.5
    s = m1**2 + m2**2 + 2 * (E1 * E2 - p(E1, m1) * p(E2, m2) * costh)
    return s
