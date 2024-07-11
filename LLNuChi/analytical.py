import numpy as np

from utils.kinematics import kallen

# cross sections
# see appendix (eqnA3, eqnA5)
def sigma_ann_V(s, mL, mChi, Lambda):
    factor1 = (1 - 4 * mL**2 / s) ** 0.5
    factor2 = 1 / (12 * np.pi * Lambda**4 * s**2 * (s - 4 * mL**2))
    factor3 = (s + 2 * mL**2) * (s - mChi**2) ** 2 * (2 * s + mChi**2)
    result = factor1 * factor2 * factor3
    return result


def sigma_scat_V(s, mL, mChi, Lambda):
    factor1 = (mL**4 - 2 * mL**2 * (s + mChi**2) + (s - mChi**2) ** 2) ** 0.5
    factor2 = 1 / (24 * np.pi * Lambda**4 * s**3)
    factor3 = (
        8 * s**3
        - s**2 * (12 * mL**2 + 7 * mChi**2)
        - s * (mChi**4 - 6 * mL**4 + 3 * mL**2 * mChi**2)
        - 2 * mL**2 * (mL**2 - mChi**2) ** 2
    )
    result = factor1 * factor2 * factor3
    return result


# TODO: more sigma's

sigmas = {"annihilation": {"V": sigma_ann_V}, "scattering": {"V": sigma_scat_V}}


def get_sigma(process, operator, **kwargs):
    result = sigmas[process][operator](**kwargs)
    assert np.isfinite(result).all()
    return result


# J functions
# see section 2 (eqn 19, eqn24-28)
def J_ann_V(s, mL, mChi, Lambda, E1, E2, Fdeg):
    factor1 = (E1 + E2) / (12 * np.pi * s**3 * Lambda**4)
    factor2 = (
        (s - mChi**2) ** 2 * (s + mChi**2) * (2 * s + mChi**2) * (s + 2 * mL**2)
    )
    result = factor1 * Fdeg * factor2
    return result


def J_scat_V(s, mL, mChi, Lambda, E1, E2, Fdeg):
    """# expression has typos
    factor1 = kallen(s, mL**2, mChi**2)**0.5 / (24 * np.pi * s**4 * Lambda**4)
    line1 = (
        s**4 * (3 * mChi**2 - 27 * mL**2)
        + s**3 * (-6 * mL**2 * mChi**2)
        + 42 * mL**4
        - 9 * mChi**4
    )
    line2 = s**2 * (
        20 * mL**4 * mChi**2 + 5 * mL**2 * mChi**4 - 34 * mL**6 - mChi**6
    )
    line3 = (
        s
        * (mL**2 - mChi**2)
        * (15 * mL**6 - 11 * mL**4 * mChi**2 + 2 * mL**2 * mChi**4)
    )
    line4 = -3 * mL**4 * (mL**2 - mChi**2) ** 3 + 7 * s**5
    term1 = E1 * (line1 + line2 + line3 + line4)
    line5 = (
        s**4 * (-27 * mL**2 - mChi**2)
        + s**3 * (-16 * mL**2 * mChi**2 + 34 * mL**2 - 7 * mChi**2)
        + s**2
        * (
            12 * mL**4 * mChi**2
            + 5 * mL**2 * mChi**4
            - 26 * mL**6
            - mChi**6
        )
    )
    line6 = (
        s * (mL**2 - mChi**2) * (13 * mL**6 - 7 * mL**4 * mChi**2)
        - 3 * mL**4 * (mL**2 - mChi**2) ** 3
        + 9 * s**5
    )
    term2 = E2 * (line5 + line6)
    result = factor1 * Fdeg * (term1 + term2)
    """
    result = s**2 / (24 * np.pi * Lambda**4) * (7 * E1 + 9 * E2)  # massless limit
    return result


Js = {"annihilation": {"V": J_ann_V}, "scattering": {"V": J_scat_V}}


def get_J(process, operator, **kwargs):
    result = Js[process][operator](**kwargs)
    assert np.isfinite(result).all()
    return result
