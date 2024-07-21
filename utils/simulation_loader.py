import os
import numpy as np
import utils.constants as c

SIM_PATH = "simulations"
SIM_FILES = {
    "SFHo-18.6": "hydro-SFHo-s18.6-MUONS-T=0.99948092.txt",
    "SFHo-18.80": "hydro-SFHo-s18.80-MUONS-T=1.0001996.txt",
    "SFHo-20.0": "hydro-SFHo-s20.0-MUONS-T=1.0010874.txt",
    "LS220": "hydro-LS220-T=1.0001464.txt",
}
RAFFELT_BOUND = {
    "SFHo-18.80": 5.68e52,
    "SFHo-20.0": 1.0e53,
}  # in erg/s


def load_simulation(name):
    file = os.path.join(SIM_PATH, SIM_FILES[name])
    assert os.path.exists(file)

    tab = np.loadtxt(file, skiprows=5, dtype=np.float64)

    R = 1e-5 * tab[:, 0] * c.km2invMeV  # in MeV^-1
    T = tab[:, 4]  # in MeV
    mu = {
        "e": tab[:, 9],  # in MeV
        "mu": tab[:, 12],  # in MeV
        "p": tab[:, 11] + c.mp,
        "n": tab[:, 10] + c.mn,
        "nu_e": tab[:, 8],
        "nu_mu": tab[:, 13],
    }
    return R, T, mu


def get_RaffeltBound(name):
    try:
        bound = RAFFELT_BOUND[name]
    except KeyError:
        print(f"Key {name} not in dict. " f"Available keys: {RAFFELT_BOUND.keys()}")
    bound *= c.erg2MeV * c.invs2MeV
    return bound
