import numpy as np
import vegas

from utils.kinematics import get_s
from utils.thermodynamics import get_Fdeg
from utils.trapping import Trapper
from LLNuChi.analytical import get_sigma


class Trapper_LLNuChi(Trapper):
    def get_inverse_mean_free_path(self, operator, R, **kwargs):
        if self.approach == "inverse":
            imfp = 0.0
            imfp += self._get_imfp_inverse_ann(operator, **kwargs)
            imfp += self._get_imfp_inverse_scat(operator, antilepton=False, **kwargs)
            imfp += self._get_imfp_inverse_scat(operator, antilepton=True, **kwargs)
        elif self.approach == "exact":
            raise NotImplementedError
        else:
            raise ValueError(f"approach {self.approach} not implemented")
        return imfp

    def _get_imfp_inverse_ann(self, operator, T, mu, model):
        mu_L, mu_nuL = mu["e"], mu["nu_e"]
        mL, mChi, Lambda = [model[key] for key in ["mL", "mChi", "Lambda"]]
        Fdeg_Lm = get_Fdeg(mL, T, mu_L, is_boson=False)
        Fdeg_Lp = get_Fdeg(mL, T, -mu_L, is_boson=False)

        def get_integrand(s, Echi, Enu):
            sigma = get_sigma(
                "annihilation", operator, s=s, mL=mL, mChi=mChi, Lambda=Lambda
            )
            factor1 = 1 / (4 * np.pi * Echi) * (s - mChi**2)
            factor2 = Enu / (np.exp((Enu - mu_nuL) / T) + 1)
            integrand = Fdeg_Lm * Fdeg_Lp * factor1 * factor2 * sigma
            mask1 = s > 4 * mL**2
            mask2 = s > mChi**2
            mask = np.logical_and(mask1, mask2)
            integrand[~mask] = 0.0
            return integrand

        @vegas.batchintegrand
        def f(x):
            a, b, costh = x[..., 0], x[..., 1], x[..., 2]
            Echi, Enu = T * a / (1 - a), T * b / (1 - b)
            jac = T / (1 - a) ** 2 * T / (1 - b) ** 2
            s = get_s(mChi, 0.0, Echi, Enu, costh)
            integrand = get_integrand(s, Echi, Enu)
            assert np.isfinite(integrand).all()
            return integrand

        integrator = vegas.Integrator(
            [[mChi / (T + mChi), 1.0], [0.0, 1.0], [-1.0, 1.0]]
        )
        factor = integrator(f, nitn=self.nitn, neval=self.neval, alpha=self.alpha).mean
        norm = self.mean_free_path_normalization(mChi, T)
        return factor / norm

    def _get_imfp_inverse_scat(self, operator, antilepton, T, mu, model):
        mu_L, mu_nuL = -mu["e"] if antilepton else mu["e"], mu["nu_e"]
        mL, mChi, Lambda = [model[key] for key in ["mL", "mChi", "Lambda"]]
        Fdeg_L = get_Fdeg(mL, T, mu_L, is_boson=False)
        Fdeg_nu = get_Fdeg(0.0, T, mu_nuL, is_boson=False)

        def get_integrand(s, Echi, EL):
            sigma = get_sigma(
                "scattering", operator, s=s, mL=mL, mChi=mChi, Lambda=Lambda
            )
            factor1 = (
                1
                / (4 * np.pi * Echi)
                * ((s - mL**2 - mChi**2) ** 2 - (2 * mL**2 * mChi**2)) ** 0.5
            )
            factor2 = (EL**2 - mL**2) ** 0.5 / (np.exp((EL - mu_L) / T) + 1)
            integrand = Fdeg_L * Fdeg_nu * factor1 * factor2 * sigma
            mask1 = s > (mL + mChi) ** 2
            mask2 = s > mL**2
            mask = np.logical_and(mask1, mask2)
            integrand[~mask] = 0.0
            return integrand

        @vegas.batchintegrand
        def f(x):
            a, b, costh = x[..., 0], x[..., 1], x[..., 2]
            Echi, EL = T * a / (1 - a), T * b / (1 - b)
            jac = T / (1 - a) ** 2 * T / (1 - b) ** 2
            s = get_s(mChi, mL, Echi, EL, costh)
            integrand = get_integrand(s, Echi, EL)
            assert np.isfinite(integrand).all()
            return integrand

        integrator = vegas.Integrator(
            [[mChi / (T + mChi), 1.0], [mL / (T + mL), 1.0], [-1.0, 1.0]]
        )
        factor = integrator(f, nitn=self.nitn, neval=self.neval, alpha=self.alpha).mean
        norm = self.mean_free_path_normalization(mChi, T)
        return factor / norm
