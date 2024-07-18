import os, sys
import warnings

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from tqdm import tqdm

import utils.constants as c
from utils.simulation_loader import load_simulation, get_RaffeltBound
from LLNuChi.free_streaming import FreeStreamer_LLNuChi


def main(operator, sim_name, lepton="e", save=True):
    print(
        f"### Free-streaming calculation for "
        f"lepton={lepton}, operator={operator}, sim_name={sim_name} ###"
    )
    R, T, mu = load_simulation(sim_name)
    assert lepton in ["e", "mu"]
    mu["L"] = mu[lepton]
    mu["nu_L"] = mu[f"nu_{lepton}"]
    mL = c.me if lepton == "e" else c.mmu
    sim_range = [50, 80]

    mChi_min = 1e0
    mChi_max = 1e3
    n_mChi = 100
    mChi = np.logspace(np.log10(mChi_min), np.log10(mChi_max), n_mChi)

    Lambda = np.zeros_like(mChi)
    freestreamer = FreeStreamer_LLNuChi()
    for i in (pbar := tqdm(range(n_mChi), desc="")):
        model = {"mL": mL, "mChi": mChi[i], "Lambda": 1.0}
        Q = freestreamer.get_Q(operator, sim_range, R, T, mu, model=model)
        Lambda[i] = get_Lambda(Q, sim_name)
        pbar.set_description(f"mChi={mChi[i]:.2e}: Lambda={Lambda[i]:.2e}")

    results = np.stack((mChi, Lambda), axis=-1)
    if save:
        file = f"LLNuChi/results/fs_{lepton}_{operator}_{sim_name}.txt"
        np.savetxt(file, results)
        print(f"Saved results to {file}")


def get_Lambda(Q, sim_name):
    bound = get_RaffeltBound(sim_name)
    Lambda = (Q / bound) ** (1 / 4) * 1e-6  # in TeV
    return Lambda


save = True
for sim_name in ["SFHo-18.80", "SFHo-20.0"]:
    for operator in ["V", "A", "S", "P", "T"]:
        main(operator, sim_name, save=save)
