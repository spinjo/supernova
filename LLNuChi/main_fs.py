import os, sys, time
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from tqdm import tqdm

import utils.constants as c
from utils.simulation_loader import load_simulation, get_RaffeltBound
from LLNuChi.free_streaming import FreeStreamer_LLNuChi


def evaluate(operator, sim_name, lepton="e", save=True):
    print(
        f"### Free-streaming calculation for "
        f"lepton={lepton}, operator={operator}, sim_name={sim_name} ###"
    )
    R, T, mu = load_simulation(sim_name)
    assert lepton in ["e", "mu"]
    mu["L"] = mu[lepton]
    mu["nu_L"] = mu[f"nu_{lepton}"]
    mL = c.me if lepton == "e" else c.mmu
    sim_range = [
        0,
        80,
    ]  # max contribution around 50; neglected contributions at <1% level

    mChi_min = 1e0
    mChi_max = 1e3
    n_mChi = 100
    mChi = np.logspace(np.log10(mChi_min), np.log10(mChi_max), n_mChi)

    Lambda = np.zeros_like(mChi)
    freestreamer = FreeStreamer_LLNuChi()
    t0, date0 = time.time(), datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for i in (pbar := tqdm(range(n_mChi), desc="", file=sys.stdout)):
        model = {"mL": mL, "mChi": mChi[i], "Lambda": 1.0}
        Q = freestreamer.get_Q(operator, sim_range, R, T, mu, model=model)
        Lambda[i] = get_Lambda(Q, sim_name)
        pbar.set_description(f"mChi={mChi[i]:.2e}: Lambda={Lambda[i]:.2e}")
    dt, date1 = time.time() - t0, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Finished calculation after {dt:.2f}s = {dt/60:.2f}min")

    results = np.stack((mChi, Lambda), axis=-1)
    if save:
        file = f"LLNuChi/results/fs_{lepton}_{operator}_{sim_name}.txt"
        np.savetxt(
            file,
            results,
            fmt="%.2e",
            header=f"Free-streaming bound for operator={operator}, sim_name={sim_name}, lepton={lepton}\n"
            f"Duration:\t{date0} -> {date1} ({dt/60:.2f}min)\n"
            f"Parameters:\tsim_range={sim_range}, vegas_kwargs={freestreamer.vegas_kwargs}\n"
            f"m_chi [MeV] / Lambda [TeV]",
        )
        print(f"Saved results to {file}")


def get_Lambda(Q, sim_name):
    bound = get_RaffeltBound(sim_name)
    Lambda = (Q / bound) ** (1 / 4) * 1e-6  # in TeV
    return Lambda


def main(logging=True, save=True):
    # CLargument = 0..19
    CLargument = int(sys.argv[1])
    assert isinstance(CLargument, int)
    assert CLargument >= 0 and CLargument < 20
    if logging:
        sys.stdout = open(f"LLNuChi/logging/out_fs_{CLargument}.txt", "w", buffering=1)
        sys.stderr = open(f"LLNuChi/logging/err_fs_{CLargument}.txt", "w", buffering=1)

    # decode CLargument
    operator = {0: "V", 1: "A", 2: "S", 3: "P", 4: "T"}[CLargument % 5]
    CLargument = CLargument // 5
    sim_name = {0: "SFHo-18.80", 1: "SFHo-20.0"}[CLargument % 2]
    CLargument = CLargument // 2
    lepton = {0: "e", 1: "mu"}[CLargument % 2]
    CLargument = CLargument // 2
    assert CLargument == 0

    evaluate(operator, sim_name, lepton=lepton, save=save)


if __name__ == "__main__":
    main()
