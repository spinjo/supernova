import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import others.digitized_constraints as constraints
import utils.plot_settings as ps

mChi_min = {"V": 5, "S": 4}
mChi_max = {"V": 600, "S": 40}
prec = 100
mass = {
    operator: np.logspace(
        np.log10(mChi_min[operator]), np.log10(mChi_max[operator]), prec
    )
    for operator in ["V", "S"]
}
term = {
    "V": r"\bar\chi \gamma^\mu P_L\nu \bar e\gamma_\mu e",
    "S": r"\bar\chi P_L\nu \bar ee",
}

y_min = {"V": 1e-61, "S": 1e-61}
y_max = {"V": 1e-51, "S": 1e-51}
x_ticks = {"V": [10, 30, 100, 300], "S": [5, 10, 30]}

FIGSIZE = (5, 4)
LEFT, BOTTOM, RIGHT, TOP = 0.16, 0.16, 0.95, 0.95
X_LABEL_POS, Y_LABEL_POS = -0.09, -0.14

sim1, sim2 = "SFHo-18.80", "SFHo-20.0"
sim2_linestyle = (0, (1, 1))
alpha = 0.15
background_color = "grey"
lw_lines = 0.5

VELOCITY = 1e-3


def get_bounds(operator):
    bounds = constraints.get_bound_droretal(operator, VELOCITY)
    if operator in ["A", "V"]:
        x, y = (mass[operator], get_sigmav(constraints.bound_LEP[operator], operator))
        y *= VELOCITY
        bounds["LEP"] = {"x": x, "y": y}
    return bounds


def get_sigmav(Lambda, operator):
    # see https://arxiv.org/pdf/2206.02339 below eqn4
    sigma_e = mass[operator] ** 2 / (4 * np.pi * Lambda**4)
    # see https://arxiv.org/pdf/2201.11497 above eqn 2a
    # unit conversion
    # 1e-24 = (keV/GeV)^2 (GeV/TeV)^4 from converting everything to GeV
    # 4e-28 = 1e4 (5e15)^(-2) = 1e4 (GeV m)^(-2) = (GeV cm)^(-2)
    unit_conversion = 4e-28 * 1e-24  # = keV^2/TeV^4/cm^2
    return sigma_e * VELOCITY * unit_conversion


def get_SN_bounds(operator, tr_approach, lepton="e"):
    assert lepton == "e"
    bounds = {}
    for sim_name in ["SFHo-18.80", "SFHo-20.0"]:
        data_fs = np.loadtxt(f"results/fs_{lepton}_{operator}_{sim_name}.txt")
        Lambda_fs = data_fs[0, 1]
        data_tr = np.loadtxt(
            f"results/tr_{lepton}_{tr_approach}_{operator}_{sim_name}.txt"
        )
        Lambda_tr = data_tr[0, 1]
        bounds[sim_name] = {
            "x": mass[operator],
            "y_low": get_sigmav(Lambda_fs, operator),
            "y_high": get_sigmav(Lambda_tr, operator),
            "Lambda_fs": Lambda_fs,
            "Lambda_tr": Lambda_tr,
        }
    return bounds


def money_plot(tr_approach="exact"):
    filename = f"results/moneyplot_3_{tr_approach}.pdf"
    with PdfPages(filename) as file:
        for operator in ["V", "S"]:
            bounds_others = get_bounds(operator)
            bounds_SN = get_SN_bounds(operator, tr_approach)

            fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

            # axes
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(mChi_min[operator], mChi_max[operator])
            ax.set_ylim(y_min[operator], y_max[operator])
            ax.set_xlabel(r"$m_\chi$ [keV]")
            ax.set_ylabel(r"$\sigma_{\chi e} v_\chi$ [cm$^2$]")

            axr = ax.twinx()
            axr.set_yscale("log")
            axr.set_ylim(y_min[operator] / VELOCITY, y_max[operator] / VELOCITY)
            axr.set_ylabel(r"$m_\chi^2 / (4\pi \Lambda^4)$ [cm$^2$]")

            # other bounds
            x, y = [bounds_others["background"][key] for key in ["x", "y"]]
            ax.fill_between(
                x,
                y,
                y_max[operator] * np.ones_like(x),
                color=background_color,
                alpha=alpha,
            )
            ax.plot(
                bounds_others["background"]["x"],
                bounds_others["background"]["y"],
                color=background_color,
                alpha=1,
                linestyle="-",
                lw=lw_lines,
            )
            for label in ["darwin", "xenon1t"]:
                ax.plot(
                    bounds_others[label]["x"],
                    bounds_others[label]["y"],
                    color="k",
                    alpha=1.0,
                    linestyle="--",
                )

            # LEP bound
            if operator == "V":
                x, y = [bounds_others["LEP"][key] for key in ["x", "y"]]
                ax.fill_between(
                    x,
                    y,
                    y_max[operator] * np.ones_like(x),
                    alpha=alpha,
                    color=ps.colors[1],
                )
                ax.plot(x, y, color=ps.colors[1], lw=lw_lines)

            # SN bound
            x, y_low, y_high = [
                bounds_SN[sim1][key] for key in ["x", "y_low", "y_high"]
            ]
            ax.fill_between(
                x,
                y_low,
                y_high,
                alpha=alpha,
                color=ps.colors[0],
            )
            ax.plot(x, y_low, color=ps.colors[0], lw=lw_lines)
            ax.plot(x, y_high, color=ps.colors[0], lw=lw_lines)
            x, y_low, y_high = [
                bounds_SN[sim2][key] for key in ["x", "y_low", "y_high"]
            ]
            kwargs = {"linestyle": sim2_linestyle, "color": ps.colors[0]}
            ax.plot(x, y_low, **kwargs)
            ax.plot(x, y_high, **kwargs)

            ax.text(
                0.95,
                0.05,
                s=r"${%s}$" % term[operator],
                horizontalalignment="right",
                verticalalignment="bottom",
                transform=ax.transAxes,
                fontsize=1.2 * ps.FONTSIZE,
            )

            text_fields(ax, operator)

            ax.set_xticks(x_ticks[operator])
            ax.set_xticklabels(x_ticks[operator])
            ax.tick_params(axis="x", which="minor", labelbottom=False)

            ax.xaxis.set_label_coords(0.5, X_LABEL_POS)
            ax.yaxis.set_label_coords(Y_LABEL_POS, 0.5)
            plt.subplots_adjust(LEFT, BOTTOM, RIGHT, TOP)
            plt.savefig(file, bbox_inches="tight", format="pdf")
            plt.close()


def text_fields(ax, operator):
    if operator == "V":
        ax.text(
            6,
            1.8e-60,
            s="SN1987A",
            color=ps.colors[0],
            fontsize=1.2 * ps.FONTSIZE,
            rotation=19,
        )
        ax.text(
            6,
            7e-56,
            s="LEP",
            color=ps.colors[1],
            fontsize=1.2 * ps.FONTSIZE,
            rotation=19,
        )
        ax.text(
            5.1,
            3e-53,
            s="overabundance",
            color=background_color,
            alpha=1,
            fontsize=0.85 * ps.FONTSIZE,
            rotation=9,
        )
        ax.text(
            330,
            5e-55,
            s=r"$\chi\to\nu\gamma\gamma\gamma$",
            color=background_color,
            alpha=1,
            fontsize=0.85 * ps.FONTSIZE,
            rotation=-60,
        )
        ax.text(
            80,
            2e-52,
            s=r"$\chi\to\nu\nu\nu$",
            horizontalalignment="center",
            color=background_color,
            alpha=1,
            fontsize=0.85 * ps.FONTSIZE,
            rotation=0,
        )
        ax.text(
            90,
            1.7e-53,
            s=r"XENON1T",
            color="k",
            alpha=1.0,
            fontsize=1.1 * ps.FONTSIZE,
            horizontalalignment="center",
            rotation=6,
        )
        ax.text(
            90,
            9e-56,
            s=r"DARWIN",
            color="k",
            alpha=1.0,
            fontsize=1.1 * ps.FONTSIZE,
            horizontalalignment="center",
            rotation=6,
        )
    elif operator == "S":
        ax.text(
            4.3,
            1.4e-60,
            s="SN1987A",
            color=ps.colors[0],
            fontsize=1.2 * ps.FONTSIZE,
            rotation=10,
        )
        ax.text(
            8.6,
            2.3e-53,
            s=r"overabundance",
            horizontalalignment="center",
            color=background_color,
            alpha=0.5,
            fontsize=0.9 * ps.FONTSIZE,
            rotation=5,
        )
        ax.text(
            23,
            1.5e-54,
            s=r"$\chi\to\nu\gamma\gamma$",
            horizontalalignment="center",
            color=background_color,
            alpha=0.5,
            fontsize=0.9 * ps.FONTSIZE,
            rotation=-22,
        )
        ax.text(
            23,
            1.2e-52,
            s=r"XENON1T",
            color="k",
            alpha=1.0,
            fontsize=1.1 * ps.FONTSIZE,
            horizontalalignment="center",
            rotation=-8,
        )
        ax.text(
            13,
            2.3e-54,
            s=r"DARWIN",
            color="k",
            alpha=1.0,
            fontsize=1.1 * ps.FONTSIZE,
            horizontalalignment="center",
            rotation=-11,
        )


money_plot(tr_approach="inverse")
money_plot(tr_approach="exact")
