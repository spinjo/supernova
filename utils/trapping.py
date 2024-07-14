import numpy as np
from scipy.integrate import quad

from utils.supernova import get_trapping_sphere_radius


class Trapper:
    def __init__(self, mfp_prescription="thermal", approach="exact"):
        # TODO: LOGGER
        self.nitn = 10
        self.neval = 1000
        self.alpha = 0.5
        assert mfp_prescription in ["Rosseland", "thermal"]
        self.mfp_prescription = mfp_prescription
        assert approach in ["inverse", "exact"]
        self.approach = approach

    def get_opacity(self, operator, i_crit, sim_range, R, T, mu, **kwargs):
        n = sim_range
        imfp = np.zeros(n)
        for i in range(n):
            mu_i = {
                particle: mu_particle[i_crit + i]
                for particle, mu_particle in mu.items()
            }
            imfp[i] = self.get_inverse_mean_free_path(
                operator, R=R[i_crit + i], T=T[i_crit + i], mu=mu_i, **kwargs
            )
        opacity = np.trapz(imfp, x=R[i_crit : i_crit + n])
        return opacity

    def mean_free_path_weight(self, E, mass, T, is_boson=False):
        # Note: Assume that this is the mfp of the dark particle -> use mu = 0
        x = E / T
        if self.mfp_prescription == "Rosseland":
            # p**2 * E**2 * exp(x) * f**2
            weight1 = (1 - mass**2 / E**2) * x**4
            weight2 = np.where(
                E / T < 1e2,
                np.exp(x) / (np.exp(x) - 1) ** 2
                if is_boson
                else np.exp(x) / (np.exp(x) + 1) ** 2,
                np.exp(-x),
            )
            return weight1 * weight2
        elif self.mfp_prescription == "thermal":
            # E * p * f
            weight1 = (1 - mass**2 / E**2) ** 0.5 * x
            weight2 = 1 / (np.exp(x) - 1) if is_boson else 1 / (np.exp(x) + 1)
            return weight1 * weight2
        else:
            raise ValueError(
                f"mfp_prescription {self.mfp_prescription} not implemented"
            )

    def mean_free_path_normalization(self, mass, T, is_boson=False):
        norm = quad(
            lambda x: self.mean_free_path_weight(x * T, mass, T, is_boson),
            mass / T,
            np.inf,
        )[0]
        return norm

    def get_inverse_mean_free_path(self, operator, R, **kwargs):
        raise NotImplementedError
