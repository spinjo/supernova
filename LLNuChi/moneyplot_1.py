import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import others.digitized_constraints as constraints
import utils.plot_settings as ps

mChi_min = 1e0
mChi_max = 1e3
prec = 100
mass = np.logspace(np.log10(mChi_min), np.log10(mChi_max), prec)

y_min = {"V": 1e-62, "A": 1e-62, "S": 1e-62, "P": 1e-62, "T": 1e-68}
y_max = {"V": 1e-45, "A": 1e-45, "S": 1e-45, "P": 1e-45, "T": 1e-45}

FIGSIZE = (6, 5)
LEFT, BOTTOM, RIGHT, TOP = 0.16, 0.16, 0.95, 0.95
X_LABEL_POS, Y_LABEL_POS = -0.08, -0.12

sim1, sim2 = "SFHo-18.80", "SFHo-20.0"
sim2_linestyle = (0, (1, 1))


def get_bounds(operator):
    bounds = {}
    x, y, label = constraints.get_bound_overproduction(operator)
    bounds[label] = {"x": x, "y": y}
    x, y, label = constraints.get_bound_nugamma(mass, operator)
    bounds[label] = {"x": x, "y": y}
    x, y, label = constraints.get_bound_nudecay(mass, operator)
    bounds[label] = {"x": x, "y": y}
    x, y, label = constraints.get_bound_pandax(operator)
    bounds[label] = {"x": x, "y": y}
    return bounds


def get_sigmav(Lambda):
    # see https://arxiv.org/pdf/2206.02339 below eqn4
    sigma_e = mass**2 / (4 * np.pi * Lambda**4)
    # see https://arxiv.org/pdf/2201.11497 above eqn 2a
    v = 1e-3
    # unit conversion
    # 1e-24 = (keV/GeV)^2 (GeV/TeV)^4 from converting everything to GeV
    # 4e-28 = 1e4 (5e15)^(-2) = 1e4 (GeV m)^(-2) = (GeV cm)^(-2)
    unit_conversion = 4e-28 * 1e-24  # = keV^2/TeV^4/cm^2
    return sigma_e * v * unit_conversion


def get_SN_bounds(operator, tr_approach):
    bounds = {}
    for sim_name in ["SFHo-18.80", "SFHo-20.0"]:
        data_fs = np.loadtxt(f"results/fs_{operator}_{sim_name}.txt")
        Lambda_fs = data_fs[0, 1]
        data_tr = np.loadtxt(f"results/tr_{tr_approach}_{operator}_{sim_name}.txt")
        Lambda_tr = data_tr[0, 1]
        bounds[sim_name] = {
            "x": mass,
            "y_low": get_sigmav(Lambda_fs),
            "y_high": get_sigmav(Lambda_tr),
            "Lambda_fs": Lambda_fs,
            "Lambda_tr": Lambda_tr,
        }
    return bounds


def money_plot(tr_approach):
    filename = f"results/moneyplot_1_{tr_approach}.pdf"
    with PdfPages(filename) as file:
        for operator in ["V", "A", "S", "P", "T"]:
            bounds_others = get_bounds(operator)
            bounds_SN = get_SN_bounds(operator, tr_approach)

            fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

            # axes
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(mChi_min, mChi_max)
            ax.set_ylim(y_min[operator], y_max[operator])
            ax.set_xlabel(r"$m_\chi$ [keV]")
            ax.set_ylabel(r"$\sigma_{\chi e} v_\chi$ [cm$^2$]")

            # SN bound
            x, y_low, y_high = [
                bounds_SN[sim1][key] for key in ["x", "y_low", "y_high"]
            ]
            ax.fill_between(
                x,
                y_low,
                y_high,
                alpha=0.5,
                color=ps.colors[0],
                label="SN",
            )
            x, y_low, y_high = [
                bounds_SN[sim2][key] for key in ["x", "y_low", "y_high"]
            ]
            kwargs = {"linestyle": sim2_linestyle, "color": ps.colors[0], "alpha": 0.5}
            ax.plot(x, y_low, **kwargs)
            ax.plot(x, y_high, **kwargs)

            # other bounds
            unpack = lambda string: [bounds_others[key] for key in ["x", "y"]]
            for (label, value), color in zip(bounds_others.items(), ps.colors[1:]):
                x, y = [bounds_others[label][key] for key in ["x", "y"]]
                if x is None or y is None:
                    continue
                ax.fill_between(
                    x,
                    y,
                    y_max[operator] * np.ones_like(x),
                    alpha=0.5,
                    color=color,
                    label=label,
                )

            ax.text(
                0.95,
                0.95,
                s=r"${%s}$" % operator,
                horizontalalignment="right",
                verticalalignment="top",
                transform=ax.transAxes,
                fontsize=2 * ps.FONTSIZE,
            )
            ax.legend(loc=3, frameon=False)
            ax.xaxis.set_label_coords(0.5, X_LABEL_POS)
            ax.yaxis.set_label_coords(Y_LABEL_POS, 0.5)
            plt.subplots_adjust(LEFT, BOTTOM, RIGHT, TOP)
            plt.savefig(file, bbox_inches="tight", format="pdf")
            plt.close()


money_plot(tr_approach="inverse")
money_plot(tr_approach="exact")
