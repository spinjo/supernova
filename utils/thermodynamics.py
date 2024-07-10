import numpy as np
from scipy.integrate import quad


def get_f(x, mu_T, is_boson):
    # mu_T = mu/T
    if is_boson:
        return 1 / (np.exp(x - mu_T) - 1)
    else:
        return 1 / (np.exp(x - mu_T) + 1)


def get_Fdeg(mass, T, mu, is_boson):
    # x = E/T
    # note: Fdeg is dimensionless
    f = lambda x: get_f(x, mu / T, is_boson)
    weighting = lambda x: x * (x**2 - (mass / T) ** 2) ** 0.5 * f(x)
    stat_factor = lambda x: 1 + f(x) if is_boson else 1 - f(x)
    num = quad(lambda x: weighting(x) * stat_factor(x), mass / T, np.inf)[0]
    denom = quad(weighting, mass / T, np.inf)[0]
    Fdeg = num / denom
    return Fdeg


def get_n(mass, g, T, mu, is_boson):
    f = lambda x: get_f(x, mu / T, is_boson)
    fac = quad(
        lambda x: x * (x**2 - (mass / T) ** 2) ** 0.5 * f(x), mass / T, np.inf
    )[0]
    n = g / (2 * np.pi**2) * T**3 * fac
    return n
