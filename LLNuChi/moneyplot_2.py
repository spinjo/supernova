import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import utils.plot_settings as ps

mChi_min = 1e0
mChi_max = 1e3

Lambda_min = {"e": 5e-3, "mu": 2e-5}
Lambda_max = {"e": 6e1, "mu": 6e1}

FIGSIZE = (5, 3)
LEFT, BOTTOM, RIGHT, TOP = 0.16, 0.16, 0.95, 0.95
X_LABEL_POS, Y_LABEL_POS = -0.1, -0.11

sim1, sim2 = "SFHo-18.80", "SFHo-20.0"
sim2_linestyle = (0, (1, 1))
alpha = 0.7


def get_SN_bounds(operator, tr_approach, lepton):
    mass, Lambda_fs, Lambda_tr = {}, {}, {}
    for sim_name in ["SFHo-18.80", "SFHo-20.0"]:
        data = np.loadtxt(f"results/fs_{lepton}_{operator}_{sim_name}.txt")
        mass[sim_name] = data[:, 0]
        Lambda_fs[sim_name] = data[:, 1]
        data = np.loadtxt(
            f"results/tr_{lepton}_{tr_approach}_{operator}_{sim_name}.txt"
        )
        assert np.all(data[:, 0] == mass[sim_name])
        Lambda_tr[sim_name] = data[:, 1]
        nan_mask = np.isnan(Lambda_tr[sim_name])
        Lambda_fs[sim_name][nan_mask] = Lambda_tr[sim_name][nan_mask]
    return mass, Lambda_fs, Lambda_tr


def money_plot(tr_approach, lepton):
    filename = f"results/moneyplot_2_{lepton}_{tr_approach}.pdf"
    with PdfPages(filename) as file:
        for operator in ["V", "A", "S", "P", "T"]:
            mass, Lambda_fs, Lambda_tr = get_SN_bounds(
                operator, tr_approach=tr_approach, lepton=lepton
            )
            fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

            # axes
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(mChi_min, mChi_max)
            ax.set_ylim(Lambda_min[lepton], Lambda_max[lepton])
            ax.set_xlabel(r"$m_\chi$ [MeV]")
            ax.set_ylabel(r"$\Lambda_{{%s}}$ [TeV]" % operator)

            ax.fill_between(
                mass[sim1],
                Lambda_fs[sim1],
                Lambda_tr[sim1],
                alpha=alpha,
                color=ps.colors[0],
            )
            kwargs = {"linestyle": sim2_linestyle, "color": ps.colors[0], "alpha": 0.5}
            ax.plot(mass[sim2], Lambda_fs[sim2], **kwargs)
            ax.plot(mass[sim2], Lambda_tr[sim2], **kwargs)
            last_idx = np.arange(len(Lambda_tr[sim2]))[np.isfinite(Lambda_tr[sim2])][-1]
            ax.plot(
                [mass[sim2][last_idx], mass[sim2][last_idx]],
                [Lambda_fs[sim2][last_idx], Lambda_tr[sim2][last_idx]],
                **kwargs,
            )

            ax.xaxis.set_label_coords(0.5, X_LABEL_POS)
            ax.yaxis.set_label_coords(Y_LABEL_POS, 0.5)
            plt.subplots_adjust(LEFT, BOTTOM, RIGHT, TOP)
            plt.savefig(file, bbox_inches="tight", format="pdf")
            plt.close()


for tr_approach in ["exact", "inverse"]:
    for lepton in ["e", "mu"]:
        money_plot(tr_approach=tr_approach, lepton=lepton)
