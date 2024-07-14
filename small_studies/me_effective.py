import numpy as np
import matplotlib.pyplot as plt

import utils.constants as c
from utils.simulation_loader import load_simulation


def get_me_eff(T, mu):
    # following https://arxiv.org/pdf/2107.12393
    M2 = c.e**2 * (mu["e"] ** 2 + np.pi * T**2) / (8 * np.pi**2)  # eqn 8
    me_star = c.me / 2 + (c.me**2 / 4 + M2) ** 0.5  # text below eqn 8
    me_eff = 2**0.5 * me_star  # eqn 7 (also text between eqn 8 and eqn 9)
    return me_eff


def plot(sim_name):
    R, T, mu = load_simulation(sim_name)
    R = R / c.km2invMeV
    me = c.me
    me_eff = get_me_eff(T, mu)

    plt.plot(R, me + R * 0, label=r"$m_e$ [MeV]")
    plt.plot(R, me_eff, label=r"$m_{e,\mathrm{eff}}$ [MeV]")
    plt.xlim(0, 25)
    plt.ylim(0, 14)
    plt.xlabel(r"$R$ [km]")
    plt.ylabel(r"$m_e$ [MeV]")
    plt.legend(loc=1, frameon=False)
    plt.savefig("me_effective.pdf", bbox_inches="tight")
    plt.close()


plot("SFHo-18.80")
