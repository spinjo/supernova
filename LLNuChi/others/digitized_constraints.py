import numpy as np

# approximately straight lines
V_chi_to_nu3gamma = {"a": -11.37, "b": -20.94}
A_chi_to_3nu = {"a": -3.09, "b": -43.71}
V_chi_to_3nu = {"a": -3.19, "b": -43.14}


def get_bound_straight(mass, name):
    if name == "v_nu3gamma":
        a, b = [V_chi_to_nu3gamma[key] for key in ["a", "b"]]
    elif name == "a_3nu":
        a, b = [A_chi_to_3nu[key] for key in ["a", "b"]]
    elif name == "v_3nu":
        a, b = [V_chi_to_3nu[key] for key in ["a", "b"]]
    else:
        raise ValueError(f"{name} not implemented")
    x = np.log10(mass)
    y = a * x + b
    result = 10**y
    return mass, result


def get_bound_overproduction(operator):
    # operator: a, p, s, t, v
    data = np.loadtxt(f"others/overproduction_{operator}.csv", delimiter=",")
    x = data[:, 0]  # mass in keV
    y = data[:, 1]  # sigma v in cm^2
    return x, y


def get_bound_nugamma(operator):
    # operator: a, p, s, t
    data = np.loadtxt(f"others/{operator}_chi_nu2gamma.csv", delimiter=",")
    x = data[:, 0]  # mass in keV
    y = data[:, 1]  # sigma v in cm^2
    return x, y


def get_bound_pandax(operator):
    # operator: a, v
    data = np.loadtxt(f"others/pandax_{operator}.csv", delimiter=",")
    x = data[:, 0]  # mass in keV
    y = data[:, 1]  # sigma v in cm^2
    return x, y
