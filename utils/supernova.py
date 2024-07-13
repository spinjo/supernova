import numpy as np
from scipy.integrate import quad

from utils.simulation_loader import load_simulation, get_RaffeltBound


def get_trapping_luminosity_single(mChi, R, T):
    x_chi = mChi / T
    fac = 7 * np.pi**4 / 120  # mChi=0 solution
    if x_chi > 1e-2:
        fac = quad(
            lambda x: x**2 * (x**2 - x_chi**2) ** 0.5 / (np.exp(x) + 1),
            x_chi,
            np.inf,
        )[0]
    lumi = 2 / np.pi * R**2 * T**4 * fac
    return lumi


def get_trapping_luminosity_full(mChi, R, T):
    lumi = np.ones(len(R))
    for i in range(len(R)):
        lumi[i] = get_trapping_luminosity_single(mChi, R[i], T[i])
    return lumi


def get_trapping_sphere_radius(mChi, sim_name):
    R, T, _ = load_simulation(sim_name)
    bound = get_RaffeltBound(sim_name)
    lumi = get_trapping_luminosity_full(mChi, R, T)
    i_crit = len(R) - 1

    while i_crit > 0:
        if lumi[i_crit] < bound:
            i_crit = i_crit - 1
        else:
            break
    if i_crit > 0:
        return i_crit
    else:
        # particle never reaches sufficient blackbody luminosity
        # because it is too heavy
        return None
