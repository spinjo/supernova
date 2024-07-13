import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib

import others.digitized_constraints as constraints

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Charter"
plt.rcParams["text.usetex"] = True
plt.rcParams[
    "text.latex.preamble"
] = r"\usepackage[bitstream-charter]{mathdesign} \usepackage{amsmath}"

mChi_min = 1e0
mChi_max = 1e3

Lambda_min = 1e-2
Lambda_max = 1e2


def get_SN_bounds(operator):
    bounds = {}
    # free-streaming
    data = np.loadtxt(f"results/fs_V_SFHo-18.80.txt")
    mass = data[:, 0]
    Lambda_fs = data[:, 1]
    Lambda_tr = Lambda_min * np.ones_like(Lambda_fs)
    return mass, Lambda_fs, Lambda_tr


def money_plot():
    filename = "results/moneyplot_2.pdf"
    with PdfPages(filename) as file:
        for operator in ["V", "A", "S", "P", "T"]:
            mass, Lambda_fs, Lambda_tr = get_SN_bounds(operator)
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))

            # axes
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(mChi_min, mChi_max)
            ax.set_ylim(Lambda_min, Lambda_max)
            ax.set_xlabel(r"$m_\chi$ [MeV]")
            ax.set_ylabel(r"$\Lambda_{{%s}}$ [TeV]" % operator)

            ax.fill_between(mass, Lambda_fs, Lambda_tr, alpha=0.5)

            plt.savefig(file, bbox_inches="tight", format="pdf")
            plt.close()


money_plot()
