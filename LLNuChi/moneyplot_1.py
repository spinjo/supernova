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
prec = 100
mass = np.logspace(np.log10(mChi_min), np.log10(mChi_max), prec)

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
    x, y = constraints.get_bound_straight(mass, "v_nu3gamma")
    bounds[f"nugamma_v"] = {"x": x, "y": y}
    for operator in ["v", "a"]:
        x, y = constraints.get_bound_straight(mass, f"{operator}_3nu")
        bounds[f"3nu_{operator}"] = {"x": x, "y": y}
    return bounds


def get_sigmav(mass, Lambda):
    # see https://arxiv.org/pdf/2206.02339 below eqn4
    sigma_e = mass**2 / (4 * np.pi * Lambda**4)
    # see https://arxiv.org/pdf/2201.11497 above eqn 2a
    v = 1e-3
    # unit conversion
    # 1e-24 = (keV/GeV)^2 (GeV/TeV)^4 from converting everything to GeV
    # 0.25e-26 = (5e15)^(-2) = (GeV * cm)^(-2)
    unit_conversion = 0.25e-26 * 1e-24  # = keV^2/TeV^4/cm^2
    return sigma_e * v * unit_conversion


def get_SN_bounds():
    bounds = {}
    # free-streaming
    for operator in ["V"]:
        data = np.loadtxt(f"results/fs_V_SFHo-18.80.txt")
        Lambda = data[0, 1]
        bounds[operator] = {
            "x": mass,
            "y_low": get_sigmav(mass, Lambda),
            "y_high": get_sigmav(mass, Lambda / 100),
        }
    return bounds


def money_plot():
    # only V for now

    bounds = get_bounds()
    bounds_SN = get_SN_bounds()

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
    x, y_low, y_high = [bounds_SN["V"][key] for key in ["x", "y_low", "y_high"]]
    ax.fill_between(x, y_low, y_high, alpha=0.5, label="SN")

    ax.legend(loc=4, frameon=False)
    ax.set_title("Constraints on V interactions")
    plt.savefig("results/moneyplot_1.pdf", bbox_inches="tight")
    plt.close()


money_plot()
