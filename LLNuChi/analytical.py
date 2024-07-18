import numpy as np

from utils.kinematics import kallen

# cross sections
# see appendix (eqnA3, eqnA5)
def sigma_ann(operator, s, mL, mChi, Lambda):
    # initial-state-summed cross section for L+ L- > chi nu (inverse process of what we need for trapping)
    factor1 = (1 - 4 * mL**2 / s) ** 0.5
    if operator == "V":
        factor2 = 1 / (12 * np.pi * Lambda**4 * s**2 * (s - 4 * mL**2))
        factor3 = (s + 2 * mL**2) * (s - mChi**2) ** 2 * (2 * s + mChi**2)
    elif operator == "A":
        factor2 = 1 / (12 * np.pi * Lambda**4 * s**2 * (s - 4 * mL**2))
        factor3 = (s - mChi**2) ** 2 * (
            s * (2 * s + mChi**2) + 2 * mL**2 * (mChi**2 - 4 * s)
        )
    elif operator == "S":
        factor2 = 1 / (8 * np.pi * Lambda**4 * s)
        factor3 = (s - mChi**2) ** 2
    elif operator == "P":
        factor2 = 1 / (8 * np.pi * Lambda**4 * (s - 4 * mL**2))
        factor3 = (s - mChi**2) ** 2
    elif operator == "T":
        factor2 = 1 / (3 * np.pi * Lambda**4 * s**2 * (s - 4 * mL**2))
        factor3 = (s + 2 * mL**2) * (s - mChi**2) ** 2 * (s + 2 * mChi**2)
    else:
        raise ValueError(f"operator {operator} not implemented")
    result = factor1 * factor2 * factor3

    # apply correction factor to swap initial and final state
    result *= s * (s - 4 * mL**2) / (s - mChi**2) ** 2
    return result


def sigma_scat(operator, s, mL, mChi, Lambda):
    # initial-state-summed cross section for L nu > L chi (inverse process of what we need for trapping)
    factor1 = (mL**4 - 2 * mL**2 * (s + mChi**2) + (s - mChi**2) ** 2) ** 0.5
    if operator == "V":
        factor2 = 1 / (24 * np.pi * Lambda**4 * s**3)
        factor3 = (
            8 * s**3
            - s**2 * (12 * mL**2 + 7 * mChi**2)
            - s * (mChi**4 - 6 * mL**4 + 3 * mL**2 * mChi**2)
            - 2 * mL**2 * (mL**2 - mChi**2) ** 2
        )
    elif operator == "A":
        factor2 = 1 / (24 * np.pi * Lambda**4 * s**3)
        factor3 = (
            8 * s**3
            - s**2 * 7 * mChi**2
            - s * (mChi**4 + 6 * mL**4 - 9 * mL**2 * mChi**2)
            - 2 * mL**2 * (mL**2 - mChi**2) ** 2
        )
    elif operator == "S":
        factor2 = 1 / (48 * np.pi * Lambda**4 * s**3)
        factor3 = (
            2 * s**3
            - s**2 * (mChi**2 - 6 * mL**2)
            - s * (6 * mL**4 + mChi**4 - 9 * mChi**2 * mL**2)
            - 2 * mL**2 * (mL**2 - mChi**2) ** 2
        )
    elif operator == "P":
        factor2 = 1 / (48 * np.pi * Lambda**4 * s**3)
        factor3 = (
            2 * s**3
            - s**2 * (6 * mL**2 + mChi**2)
            - s * (mChi**4 - 6 * mL**4 + 3 * mL**2 * mChi**2)
            - 2 * mL**2 * (mL**2 - mChi**2) ** 2
        )
    elif operator == "T":
        factor2 = 1 / (6 * np.pi * Lambda**4 * s**3)
        factor3 = (
            14 * s**3
            - s**2 * (12 * mL**2 + 13 * mChi**2)
            - s * mChi**2 * (mChi**2 - 3 * mL**2)
            - 2 * mL**2 * (mL**2 - mChi**2) ** 2
        )
    else:
        raise ValueError(f"operator {operator} not implemented")
    result = factor1 * factor2 * factor3

    # apply correction factor to swap initial and final state
    result *= (s - mL**2) ** 2 / (
        s**2 - 2 * s * (mL**2 + mChi**2) + (mL**2 - mChi**2) ** 2
    )
    return result


sigmas = {"annihilation": sigma_ann, "scattering": sigma_scat}


def get_sigma(process, operator, **kwargs):
    return sigmas[process](operator, **kwargs)


# J functions
# note: J functions are summed over initial and final state
# see section 2 (eqn 19, eqn24-28)
def J_ann(operator, s, mL, mChi, Lambda, E1, E2, Fdeg):
    if operator == "V":
        factor1 = (E1 + E2) / (12 * np.pi * s**3 * Lambda**4)
        factor2 = (
            (s - mChi**2) ** 2
            * (s + mChi**2)
            * (2 * s + mChi**2)
            * (s + 2 * mL**2)
        )
    elif operator == "A":
        factor1 = (E1 + E2) / (12 * np.pi * s**3 * Lambda**4)
        factor2 = (
            (s - mChi**2) ** 2
            * (s + mChi**2)
            * (2 * mL**2 * (mChi**2 - 4 * s) + s * (mChi**2 + 2 * s))
        )
    elif operator == "S":
        factor1 = (E1 + E2) / (8 * np.pi * s**2 * Lambda**4)
        factor2 = (s - mChi**2) ** 2 * (s + mChi**2) * (s - 4 * mL**2)
    elif operator == "P":
        factor1 = (E1 + E2) / (12 * np.pi * s * Lambda**4)
        factor2 = (s - mChi**2) ** 2 * (s + mChi**2)
    elif operator == "T":
        factor1 = (E1 + E2) / (3 * np.pi * s**3 * Lambda**4)
        factor2 = (
            (s - mChi**2) ** 2
            * (s + mChi**2)
            * (s + 2 * mL**2)
            * (s + 2 * mChi**2)
        )
    else:
        raise ValueError(f"operator {operator} not implemented")
    result = factor1 * factor2 * Fdeg
    return result


def J_scat(operator, s, mL, mChi, Lambda, E1, E2, Fdeg):
    if operator == "V":
        factor1 = kallen(s, mL**2, mChi**2) ** 0.5 / (
            24 * np.pi * s**4 * Lambda**4
        )
        line1 = (
            s**4 * (3 * mChi**2 - 27 * mL**2)
            + s**3 * (-6 * mL**2 * mChi**2)
            + 42 * mL**4
            - 9 * mChi**4
        )
        line2 = s**2 * (
            20 * mL**4 * mChi**2
            + 5 * mL**2 * mChi**4
            - 34 * mL**6
            - mChi**6
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
        result = factor1 * (term1 + term2)
        # result = s**2 / (24 * np.pi * Lambda**4) * (7 * E1 + 9 * E2)  # massless limit
    elif operator == "A":
        factor1 = kallen(s, mL**2, mChi**2) ** 0.5 / (
            24 * np.pi * s**4 * Lambda**4
        )
        line1 = s**4 * (3 * mChi**2 - 27 * mL**2) + s**3 * (
            -6 * mL**2 * mChi**2 + 42 * mL**4 - 9 * mChi**4
        )
        line2 = s**2 * (
            -28 * mL**4 * mChi**2
            + 21 * mL**2 * mChi**4
            + 14 * mL**6
            - mChi**6
        ) + s * (
            6 * mL**6 * mChi**2
            - 3 * mL**4 * mChi**4
            - 2 * mL**2 * mChi**6
            - mL**8
        )
        line3 = (
            9 * mL**8 * mChi**2
            - 9 * mL**6 * mChi**4
            + 3 * mL**4 * mChi**6
            - 3 * mL**10
            + 7 * s**5
        )
        term1 = E1 * (line1 + line2 + line3)
        line4 = (
            s**4 * (-19 * mL**2 - mChi**2)
            + s**3 * (16 * mL**2 * mChi**2 + 2 * mL**4 - 7 * mChi**4)
            + s**2
            * (
                -20 * mL**4 * mChi**2
                + 13 * mL**2 * mChi**4
                + 14 * mL**6
                - mChi**6
            )
        )
        line5 = (
            -3 * s * mL**4 * (-4 * mL**2 * mChi**2 + mL**4 + 3 * mChi**4)
            - 3 * mL**4 * (mL**2 - mChi**2) ** 3
            + 9 * s**5
        )
        term2 = E2 * (line4 + line5)
        result = factor1 * (term1 + term2)
        # result = s**2 / (24 * np.pi * Lambda**4) * (7 * E1 + 9 * E2)  # massless limit
    elif operator == "S":
        factor1 = kallen(s, mL**2, mChi**2) ** 0.5 / (
            48 * np.pi * s**4 * Lambda**4
        )
        line1 = s**4 * (mL**2 - mChi**2) + s**3 * (
            10 * mL**2 * mChi**2 - 18 * mL**4 - mChi**4
        )
        line2 = s**2 * (
            -24 * mL**4 * mChi**2
            + 13 * mL**2 * mChi**4
            + 18 * mL**6
            - mChi**6
        )
        line3 = (
            s
            * (
                -(mL**6) * (mL**2 - mChi**2)
                + 5 * mL**4 * mChi**2 * (mL**2 - mChi**2)
                + 2 * mL**2 * mChi**4 * (mL**2 - mChi**2)
            )
            - 3 * mL**4 * (mL**2 - mChi**2) ** 3
            + 3 * s**5
        )
        term1 = E1 * (line1 + line2 + line3)
        line4 = (
            s**4 * (mL**2 + 3 * mChi**2)
            + s**3 * (24 * mL**2 * mChi**2 - 14 * mL**4 - 3 * mChi**4)
            + s**2
            * (
                -16 * mL**4 * mChi**2
                + 5 * mL**2 * mChi**4
                + 18 * mL**6
                - mChi**6
            )
        )
        line5 = (
            s
            * (
                9 * mL**4 * mChi**2 * (mL**2 - mChi**2)
                - 3 * mL**6 * (mL**2 - mChi**2)
            )
            - 3 * mL**4 * (mL**2 - mChi**2) ** 3
            + s**5
        )
        term2 = E2 * (line4 + line5)
        result = factor1 * (term1 + term2)
        # result = s**2 / (48 * np.pi * Lambda**4) * (3*E1 + E2)  # massless limit
    elif operator == "P":
        factor1 = kallen(s, mL**2, mChi**2) ** 0.5 / (
            48 * np.pi * s**4 * Lambda**4
        )
        line1 = s**4 * (-15 * mL**2 - mChi**2) + s**3 * (
            -6 * mL**2 * mChi**2 + 30 * mL**4 - mChi**4
        )
        line2 = s**2 * (
            24 * mL**4 * mChi**2
            - 3 * mL**2 * mChi**4
            - 30 * mL**6
            - mChi**6
        )
        line3 = (
            s
            * (
                15 * mL**6 * (mL**2 - mChi**2)
                - 11 * mL**4 * mChi**2 * (mL**2 - mChi**2)
                + 2 * mL**2 * mChi**4 * (mL**2 - mChi**2)
            )
            - 3 * mL**4 * (mL**2 - mChi**2) ** 3
            + 3 * s**5
        )
        term1 = E1 * (line1 + line2 + line3)
        line4 = (
            s**4 * (3 * mChi**2 - 7 * mL**2)
            + s**3 * (-8 * mL**2 * mChi**2 + 18 * mL**4 - 3 * mChi**4)
            + s**2
            * (
                16 * mL**4 * mChi**2
                - 3 * mL**2 * mChi**4
                - 22 * mL**6
                - mChi**6
            )
        )
        line5 = (
            s
            * (
                13 * mL**6 * (mL**2 - mChi**2)
                - 7 * mL**4 * mChi**2 * (mL**2 - mChi**2)
            )
            - 3 * mL**4 * (mL**2 - mChi**2)
            + s**5
        )
        term2 = E2 * (line4 + line5)
        result = factor1 * (term1 + term2)
        # result = s**2 / (48 * np.pi * Lambda**4) * (3*E1 + E2)  # massless limit
    elif operator == "T":
        factor1 = kallen(s, mL**2, mChi**2) ** 0.5 / (
            24 * np.pi * s**4 * Lambda**4
        )
        line1 = 4 * s**4 * (7 * mChi**2 - 31 * mL**2) + 4 * s**3 * (
            2 * mL**2 * mChi**2 + 30 * mL**4 - 17 * mChi**4
        )
        line2 = (
            4
            * s**2
            * (
                -8 * mL**4 * mChi**2
                + 21 * mL**2 * mChi**4
                - 14 * mL**6
                - mChi**6
            )
        )
        line3 = (
            4
            * s
            * (
                7 * mL**6 * (mL**2 - mChi**2)
                - 3 * mL**4 * mChi**2 * (mL**2 - mChi**2)
                + 2 * mL**2 * mChi**4 * (mL**2 - mChi**2)
            )
            - 12 * mL**4 * (mL**2 - mChi**2) ** 3
            + 44 * s**5
        )
        term1 = E1 * (line1 + line2 + line3)
        line4 = (
            4 * s**4 * (-43 * mL**2 - 5 * mChi**2)
            + 4 * s**3 * (-8 * mL**2 * mChi**2 + 34 * mL**4 - 11 * mChi**4)
            + 4
            * s**2
            * (
                -8 * mL**4 * mChi**2
                + 17 * mL**2 * mChi**4
                - 10 * mL**6
                - mChi**6
            )
        )
        line5 = (
            4
            * s
            * (
                5 * mL**6 * (mL**2 - mChi**2)
                + mL**4 * mChi**2 * (mL**2 - mChi**2)
            )
            - 12 * mL**4 * (mL**2 - mChi**2) ** 3
            + 68 * s**5
        )
        term2 = E2 * (line4 + line5)
        result = factor1 * (term1 + term2)
        # result = s**2 / (6 * np.pi * Lambda**4) * (11*E1 + 17*E2)  # massless limit
    else:
        raise ValueError(f"operator {operator} not implemented")
    return result * Fdeg


Js = {"annihilation": J_ann, "scattering": J_scat}


def get_J(process, operator, **kwargs):
    return Js[process](operator, **kwargs)
