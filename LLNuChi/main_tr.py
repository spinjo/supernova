import os, sys
import warnings

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from tqdm import tqdm

import utils.constants as c
from utils.simulation_loader import load_simulation, get_RaffeltBound
from LLNuChi.trapping import Trapper_LLNuChi
from utils.supernova import get_trapping_sphere_radius


def main(operator, sim_name, approach, save=True):
    print(
        f"### Trapping calculation for " f"operator={operator}, sim_name={sim_name} ###"
    )
    R, T, mu = load_simulation(sim_name)
    sim_range = 10

    mChi_min = 1e0
    mChi_max = 1e3
    n_mChi = 100
    mChi = np.logspace(np.log10(mChi_min), np.log10(mChi_max), n_mChi)

    Lambda = np.zeros_like(mChi)
    trapper = Trapper_LLNuChi(approach=approach)
    for i in (pbar := tqdm(range(n_mChi), desc="")):
        model = {"mL": c.me, "mChi": mChi[i], "Lambda": 1.0}
        i_crit = get_trapping_sphere_radius(mChi[i], sim_name)
        if i_crit is None:
            Lambda[i] = None
        else:
            opacity = trapper.get_opacity(
                operator, i_crit, sim_range, R, T, mu, model=model
            )
            Lambda[i] = get_Lambda(opacity)
        pbar.set_description(f"mChi={mChi[i]:.2e}: Lambda={Lambda[i]:.2e}")

    results = np.stack((mChi, Lambda), axis=-1)
    file = f"LLNuChi/results/tr_{operator}_{sim_name}.txt"
    np.savetxt(file, results)
    print(f"Saved results to {file}")


def get_Lambda(opacity):
    Lambda = (opacity / c.two_thirds) ** (1 / 4) * 1e-6
    return Lambda


save = True
approach = "inverse"
main("V", "SFHo-18.80", approach, save=save)
main("A", "SFHo-18.80", approach, save=save)
main("S", "SFHo-18.80", approach, save=save)
main("P", "SFHo-18.80", approach, save=save)
main("T", "SFHo-18.80", approach, save=save)
