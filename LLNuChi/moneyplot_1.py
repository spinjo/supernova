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

mChi_min = 2
mChi_max = 900

y_min = 1e-55
y_max = 1e-45


def get_bounds():
    bounds = {}
    for operator in ["s", "p", "v", "a", "t"]:
        x, y = constraints.get_bound_overproduction(operator)
        bounds[f"overproduction_{operator}"] = {"x": x, "y": y}
    for operator in ["s", "p", "t", "a"]:
        x, y = constraints.get_bound_nugamma(operator)
        bounds[f"nugamma_{operator}"] = {"x": x, "y": y}
    for operator in ["v", "a"]:
        x, y = constraints.get_bound_pandax(operator)
        bounds[f"pandax_{operator}"] = {"x": x, "y": y}
    prec = 100
    mass = np.logspace(np.log10(mChi_min), np.log10(mChi_max), prec)
    x, y = constraints.get_bound_straight(mass, "v_nu3gamma")
    bounds[f"nugamma_v"] = {"x": x, "y": y}
    for operator in ["v", "a"]:
        x, y = constraints.get_bound_straight(mass, f"{operator}_3nu")
        bounds[f"3nu_{operator}"] = {"x": x, "y": y}
    return bounds


def money_plot():
    # only V for now

    bounds = get_bounds()

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # axes
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(mChi_min, mChi_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(r"$m_\chi$ [keV]")
    ax.set_ylabel(r"$\sigma_{\chi e} v_\chi$ [cm$^2$]")

    # contents
    unpack = lambda string: [bounds[string][key] for key in ["x", "y"]]
    for string in ["pandax", "overproduction", "nugamma", "3nu"]:
        x, y = unpack(f"{string}_v")
        ax.fill_between(x, y, y_max * np.ones_like(x), alpha=0.5, label=string)

    ax.legend(loc=3, frameon=False)
    ax.set_title("Constraints on V interactions")
    plt.savefig("results/moneyplot_1.pdf", bbox_inches="tight")
    plt.close()


money_plot()
