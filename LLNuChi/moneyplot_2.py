import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import utils.plot_settings as ps

mChi_min = 1e0
mChi_max = 1e3

Lambda_min = 1.2e-2
Lambda_max = 3.5e1

FIGSIZE = (5, 3)
LEFT, BOTTOM, RIGHT, TOP = 0.16, 0.16, 0.95, 0.95
X_LABEL_POS, Y_LABEL_POS = -0.1, -0.11


def get_SN_bounds(operator, tr_approach):
    bounds = {}
    # free-streaming
    data = np.loadtxt(f"results/fs_{operator}_SFHo-18.80.txt")
    mass = data[:, 0]
    Lambda_fs = data[:, 1]
    data = np.loadtxt(f"results/tr_{tr_approach}_{operator}_SFHo-18.80.txt")
    assert np.all(data[:, 0] == mass)
    Lambda_tr = data[:, 1]
    return mass, Lambda_fs, Lambda_tr


def money_plot(tr_approach):
    filename = f"results/moneyplot_2_{tr_approach}.pdf"
    with PdfPages(filename) as file:
        for operator in ["V", "A", "S", "P", "T"]:
            mass, Lambda_fs, Lambda_tr = get_SN_bounds(operator, tr_approach)
            fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

            # axes
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(mChi_min, mChi_max)
            ax.set_ylim(Lambda_min, Lambda_max)
            ax.set_xlabel(r"$m_\chi$ [MeV]")
            ax.set_ylabel(r"$\Lambda_{{%s}}$ [TeV]" % operator)

            ax.fill_between(mass, Lambda_fs, Lambda_tr, alpha=0.5, color=ps.colors[0])
            ax.xaxis.set_label_coords(0.5, X_LABEL_POS)
            ax.yaxis.set_label_coords(Y_LABEL_POS, 0.5)
            plt.subplots_adjust(LEFT, BOTTOM, RIGHT, TOP)
            plt.savefig(file, bbox_inches="tight", format="pdf")
            plt.close()


money_plot(tr_approach="inverse")
money_plot(tr_approach="exact")
