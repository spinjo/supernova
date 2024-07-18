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
    # note: Factors of g cancel
    x_mass = mass / T
    f = lambda x: get_f(x, mu / T, is_boson)
    weighting = lambda x: x * (x**2 - x_mass**2) ** 0.5 * f(x)
    stat_factor = lambda x: 1 + f(x) if is_boson else 1 - f(x)
    num = quad(lambda x: weighting(x) * stat_factor(x), x_mass, np.inf)[0]
    denom = quad(weighting, x_mass, np.inf)[0]
    Fdeg = num / denom
    return Fdeg
