import numpy as np

# approximately straight lines
V_chi_to_nu3gamma = {"a": -11.37, "b": -20.94}
A_chi_to_3nu = {"a": -3.09, "b": -43.71}
V_chi_to_3nu = {"a": -3.19, "b": -43.14}

# LEP bound by Claudio (on Lambda, in TeV)
bound_LEP = {"V": 0.4, "A": 0.37}


def get_bound_straight(mass, name):
    if name == "V_nu3gamma":
        a, b = [V_chi_to_nu3gamma[key] for key in ["a", "b"]]
    elif name == "A_3nu":
        a, b = [A_chi_to_3nu[key] for key in ["a", "b"]]
    elif name == "V_3nu":
        a, b = [V_chi_to_3nu[key] for key in ["a", "b"]]
    else:
        raise ValueError(f"{name} not implemented")
    x = np.log10(mass)
    y = a * x + b
    result = 10**y
    return mass, result


def get_bound_overproduction(operator):
    assert operator in ["V", "A", "S", "P", "T"]
    data = np.loadtxt(f"others/overproduction_{operator}.csv", delimiter=",")
    x = data[:, 0]  # mass in keV
    y = data[:, 1]  # sigma v in cm^2
    return x, y, "overproduction"


def get_bound_nugamma(mass, operator):
    assert operator in ["V", "A", "S", "P", "T"]
    if operator in ["A", "P", "S", "T"]:
        data = np.loadtxt(f"others/{operator}_chi_nu2gamma.csv", delimiter=",")
        x = data[:, 0]  # mass in keV
        y = data[:, 1]  # sigma v in cm^2
    elif operator == "V":
        x, y = get_bound_straight(mass, "V_nu3gamma")
    LABEL_DICT = {
        "V": r"$\chi\to\nu\gamma\gamma\gamma$",
        "A": r"$\chi\to\nu\gamma\gamma$",
        "S": r"$\chi\to\nu\gamma\gamma$",
        "P": r"$\chi\to\nu\gamma\gamma$",
        "T": r"$\chi\to\nu\gamma$",
    }
    return x, y, LABEL_DICT[operator]


def get_bound_pandax(operator):
    if operator in ["V", "A"]:
        data = np.loadtxt(f"others/pandax_{operator}.csv", delimiter=",")
        x = data[:, 0]  # mass in keV
        y = data[:, 1]  # sigma v in cm^2
    else:
        x, y = None, None
    return x, y, "PANDAX"


def get_bound_nudecay(mass, operator):
    if operator in ["V", "A"]:
        x, y = get_bound_straight(mass, f"{operator}_3nu")
    else:
        x, y = None, None
    return x, y, r"$\chi\to 3\nu$"
