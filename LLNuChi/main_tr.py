import os, sys, time
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from tqdm import tqdm

import utils.constants as c
from utils.simulation_loader import load_simulation, get_RaffeltBound
from LLNuChi.trapping import Trapper_LLNuChi
import LLNuChi.trapping as tr
from utils.supernova import get_trapping_sphere_radius


def evaluate(operator, sim_name, approach, lepton="e", save=True):
    print(
        f"### Trapping calculation for "
        f"approach={approach}, lepton={lepton}, operator={operator}, sim_name={sim_name} ###"
    )
    R, T, mu = load_simulation(sim_name)
    assert lepton in ["e", "mu"]
    mu["L"] = mu[lepton]
    mu["nu_L"] = mu[f"nu_{lepton}"]
    mL = c.me if lepton == "e" else c.mmu
    sim_range = {"e": 100, "mu": 100}[lepton]  # neglected contributions at <10% level

    mChi_min = 1e0
    mChi_max = 1e3
    n_mChi = 100
    mChi = np.logspace(np.log10(mChi_min), np.log10(mChi_max), n_mChi)

    Lambda = np.zeros_like(mChi)
    trapper = Trapper_LLNuChi(approach=approach)
    t0, date0 = time.time(), datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for i in (pbar := tqdm(range(n_mChi), desc="", file=sys.stdout)):
        model = {"mL": mL, "mChi": mChi[i], "Lambda": 1.0}
        i_crit = get_trapping_sphere_radius(mChi[i], sim_name)
        if i_crit is None:
            Lambda[i] = None
        else:
            opacity = trapper.get_opacity(
                operator, i_crit, sim_range, R, T, mu, model=model
            )
            Lambda[i] = get_Lambda(opacity)
        pbar.set_description(f"mChi={mChi[i]:.2e}: Lambda={Lambda[i]:.2e}")
    dt, date1 = time.time() - t0, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Finished calculation after {dt:.2f}s = {dt/60:.2f}min")

    results = np.stack((mChi, Lambda), axis=-1)
    if save:
        file = f"LLNuChi/results/tr_{lepton}_{approach}_{operator}_{sim_name}.txt"
        np.savetxt(
            file,
            results,
            fmt="%.2e",
            header=f"Trapping bound for operator={operator}, sim_name={sim_name}, lepton={lepton}\n"
            f"using approach={approach}, mfp_prescription={trapper.mfp_prescription}\n"
            f"Duration:\t{date0} -> {date1} ({dt/60:.2f}min)\n"
            f"Parameters:\tsim_range={sim_range}, X_MAX={tr.X_MAX}, NSTEPS={tr.NSTEPS}, vegas_kwargs={trapper.vegas_kwargs}\n"
            f"m_chi [MeV] / Lambda [TeV]",
        )
        print(f"Saved results to {file}")


def get_Lambda(opacity):
    Lambda = (opacity / c.two_thirds) ** (1 / 4) * 1e-6
    return Lambda


def main(logging=True, save=True):
    # CLargument = 0..39
    CLargument = int(sys.argv[1])
    assert isinstance(CLargument, int)
    assert CLargument >= 0 and CLargument < 40
    if logging:
        sys.stdout = open(f"LLNuChi/logging/out_tr_{CLargument}.txt", "w", buffering=1)
        sys.stderr = open(f"LLNuChi/logging/err_tr_{CLargument}.txt", "w", buffering=1)

    # decode CLargument
    operator = {0: "V", 1: "A", 2: "S", 3: "P", 4: "T"}[CLargument % 5]
    CLargument = CLargument // 5
    sim_name = {0: "SFHo-18.80", 1: "SFHo-20.0"}[CLargument % 2]
    CLargument = CLargument // 2
    lepton = {0: "e", 1: "mu"}[CLargument % 2]
    CLargument = CLargument // 2
    approach = {0: "exact", 1: "inverse"}[CLargument % 2]
    CLargument = CLargument // 2
    assert CLargument == 0

    evaluate(operator, sim_name, approach, lepton=lepton, save=save)


if __name__ == "__main__":
    main()
