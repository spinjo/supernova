import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import numpy as np
import matplotlib.pyplot as plt

import utils.constants as c
from utils.simulation_loader import load_simulation
from utils.supernova import get_mL_eff


def plot(sim_name, lepton="e"):
    R, T, mu = load_simulation(sim_name)
    R = R / c.km2invMeV
    mL = c.me if lepton == "e" else c.mmu
    mL_eff = get_mL_eff(mL, T, mu[lepton])

    plt.plot(R, mL + R * 0, label=r"$m_e$ [MeV]")
    plt.plot(R, mL_eff, label=r"$m_{e,\mathrm{eff}}$ [MeV]")
    plt.xlim(0, 25)
    plt.xlabel(r"$R$ [km]")
    plt.ylabel(r"$m_e$ [MeV]")
    plt.legend(loc=1, frameon=False)
    plt.savefig("small_studies/me_effective.pdf", bbox_inches="tight")
    plt.close()


plot("SFHo-18.80", lepton="e")
