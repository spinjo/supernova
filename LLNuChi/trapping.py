import numpy as np
import vegas
from scipy.interpolate import interp1d
from scipy.integrate import quad

from utils.kinematics import get_s
from utils.thermodynamics import get_Fdeg
from utils.trapping import Trapper
from LLNuChi.analytical import get_sigma

GAMMA_MIN = 1e-100
X_MAX = 1e2


class Trapper_LLNuChi(Trapper):
    def get_inverse_mean_free_path(self, operator, R, **kwargs):
        if self.approach == "inverse":
            imfp = 0.0
            imfp += self._get_imfp_inverse_ann(operator, **kwargs)
            imfp += self._get_imfp_inverse_scat(operator, antilepton=False, **kwargs)
            imfp += self._get_imfp_inverse_scat(operator, antilepton=True, **kwargs)
        elif self.approach == "exact":
            imfp = self.get_imfp_exact(operator, **kwargs)
        else:
            raise ValueError(f"approach {self.approach} not implemented")
        return imfp

    def _get_gamma_diff_ann(
        self, operator, s, Echi, Enu, mL, mChi, Lambda, T, mu_nuL, Fdeg_Lm, Fdeg_Lp
    ):
        sigma = get_sigma(
            "annihilation", operator, s=s, mL=mL, mChi=mChi, Lambda=Lambda
        )
        factor1 = 1 / (4 * np.pi * Echi) * (s - mChi**2)
        factor2 = Enu / (np.exp((Enu - mu_nuL) / T) + 1)
        gamma = Fdeg_Lm * Fdeg_Lp * factor1 * factor2 * sigma
        mask1 = s > 4 * mL**2
        mask2 = s > mChi**2
        mask = np.logical_and(mask1, mask2)
        gamma[~mask] = 0.0
        return gamma

    def _get_gamma_diff_scat(
        self, operator, s, Echi, EL, mL, mChi, Lambda, T, mu_L, Fdeg_L, Fdeg_nu
    ):
        sigma = get_sigma("scattering", operator, s=s, mL=mL, mChi=mChi, Lambda=Lambda)
        factor1 = (
            1
            / (4 * np.pi * Echi)
            * ((s - mL**2 - mChi**2) ** 2 - (2 * mL**2 * mChi**2)) ** 0.5
        )
        factor2 = (EL**2 - mL**2) ** 0.5 / (np.exp((EL - mu_L) / T) + 1)
        gamma = Fdeg_L * Fdeg_nu * factor1 * factor2 * sigma
        mask1 = s > (mL + mChi) ** 2
        mask2 = s > mL**2
        mask = np.logical_and(mask1, mask2)
        gamma[~mask] = 0.0
        return gamma

    def _get_imfp_inverse_ann(self, operator, T, mu, model):
        mu_L, mu_nuL = mu["e"], mu["nu_e"]
        mL, mChi, Lambda = [model[key] for key in ["mL", "mChi", "Lambda"]]
        Fdeg_Lm = get_Fdeg(mL, T, mu_L, is_boson=False)
        Fdeg_Lp = get_Fdeg(mL, T, -mu_L, is_boson=False)

        @vegas.batchintegrand
        def f(x):
            a, b, costh = x[..., 0], x[..., 1], x[..., 2]
            Echi, Enu = T * a / (1 - a), T * b / (1 - b)
            jac = T / (1 - a) ** 2 * T / (1 - b) ** 2
            s = get_s(mChi, 0.0, Echi, Enu, costh)
            gamma = self._get_gamma_diff_ann(
                operator, s, Echi, Enu, mL, mChi, Lambda, T, mu_nuL, Fdeg_Lm, Fdeg_Lp
            )
            imfp = gamma / (1 - mChi**2 / Echi**2) ** 0.5
            weight = self.mean_free_path_weight(Echi, mChi, T, is_boson=False)
            integrand = imfp * weight
            assert np.isfinite(integrand).all()
            return integrand

        integrator = vegas.Integrator(
            [[mChi / (T + mChi), 1.0], [0.0, 1.0], [-1.0, 1.0]]
        )
        factor = integrator(f, **self.vegas_kwargs).mean
        norm = self.mean_free_path_normalization(mChi, T)
        return factor / norm

    def _get_imfp_inverse_scat(self, operator, antilepton, T, mu, model):
        mu_L, mu_nuL = -mu["e"] if antilepton else mu["e"], mu["nu_e"]
        mL, mChi, Lambda = [model[key] for key in ["mL", "mChi", "Lambda"]]
        Fdeg_L = get_Fdeg(mL, T, mu_L, is_boson=False)
        Fdeg_nu = get_Fdeg(0.0, T, mu_nuL, is_boson=False)

        @vegas.batchintegrand
        def f(x):
            a, b, costh = x[..., 0], x[..., 1], x[..., 2]
            Echi, EL = T * a / (1 - a), T * b / (1 - b)
            jac = T / (1 - a) ** 2 * T / (1 - b) ** 2
            s = get_s(mChi, mL, Echi, EL, costh)
            gamma = self._get_gamma_diff_scat(
                operator, s, Echi, EL, mL, mChi, Lambda, T, mu_L, Fdeg_L, Fdeg_nu
            )
            imfp = gamma / (1 - mChi**2 / Echi**2) ** 0.5
            weight = self.mean_free_path_weight(Echi, mChi, T, is_boson=False)
            integrand = imfp * weight
            assert np.isfinite(integrand).all()
            return integrand

        integrator = vegas.Integrator(
            [[mChi / (T + mChi), 1.0], [mL / (T + mL), 1.0], [-1.0, 1.0]]
        )
        factor = integrator(f, **self.vegas_kwargs).mean
        norm = self.mean_free_path_normalization(mChi, T)
        return factor / norm

    def get_imfp_exact(self, operator, T, mu, model, nsteps=20):
        mChi = model["mChi"]
        x_min = mChi / T * (1 + 1e-5)
        Echi = np.exp(np.linspace(np.log(x_min * T), np.log(X_MAX * T), nsteps))

        # evaluate Gamma on a grid
        Gamma = np.zeros(nsteps)
        for i in range(nsteps):
            Gamma[i] += self._get_gamma_ann(
                operator, Echi=Echi[i], T=T, mu=mu, model=model
            )
            Gamma[i] += self._get_gamma_scat(
                operator, antilepton=False, Echi=Echi[i], T=T, mu=mu, model=model
            )
            Gamma[i] += self._get_gamma_scat(
                operator, antilepton=True, Echi=Echi[i], T=T, mu=mu, model=model
            )

        # interpolate over the grid and evaluate the mfp
        Gamma = np.clip(Gamma, a_min=GAMMA_MIN, a_max=None)
        logGamma_interpolation = interp1d(
            np.log(Echi / T),
            np.log(Gamma),
            kind="linear",
            fill_value="extrapolate",
        )

        def f(x):
            Echi = x * T
            Gamma = np.exp(logGamma_interpolation(np.log(x)))
            mfp = (1 - mChi**2 / Echi**2) ** 0.5 / Gamma
            weighting = self.mean_free_path_weight(Echi, mChi, T, is_boson=False)
            integrand = mfp * weighting
            return integrand

        factor = quad(f, x_min, X_MAX)[0]

        norm = self.mean_free_path_normalization(mChi, T)
        mfp = factor / norm
        imfp = 1 / mfp
        assert np.isfinite(imfp).all()
        return imfp

    def _get_gamma_ann(self, operator, Echi, T, mu, model):
        mu_L, mu_nuL = mu["e"], mu["nu_e"]
        mL, mChi, Lambda = [model[key] for key in ["mL", "mChi", "Lambda"]]
        Fdeg_Lm = get_Fdeg(mL, T, mu_L, is_boson=False)
        Fdeg_Lp = get_Fdeg(mL, T, -mu_L, is_boson=False)

        @vegas.batchintegrand
        def f(x):
            a, costh = x[..., 0], x[..., 1]
            Enu = T * a / (1 - a)
            jac = T / (1 - a) ** 2
            s = get_s(mChi, 0.0, Echi, Enu, costh)
            gamma = self._get_gamma_diff_ann(
                operator, s, Echi, Enu, mL, mChi, Lambda, T, mu_nuL, Fdeg_Lm, Fdeg_Lp
            )
            assert np.isfinite(gamma).all()
            return gamma

        integrator = vegas.Integrator([[0.0, 1.0], [-1.0, 1.0]])
        Gamma = integrator(f, **self.vegas_kwargs).mean
        return Gamma

    def _get_gamma_scat(self, operator, antilepton, Echi, T, mu, model):
        mu_L, mu_nuL = -mu["e"] if antilepton else mu["e"], mu["nu_e"]
        mL, mChi, Lambda = [model[key] for key in ["mL", "mChi", "Lambda"]]
        Fdeg_L = get_Fdeg(mL, T, mu_L, is_boson=False)
        Fdeg_nu = get_Fdeg(0.0, T, mu_nuL, is_boson=False)

        @vegas.batchintegrand
        def f(x):
            a, costh = x[..., 0], x[..., 1]
            EL = T * a / (1 - a)
            jac = T / (1 - a) ** 2
            s = get_s(mChi, mL, Echi, EL, costh)
            gamma = self._get_gamma_diff_scat(
                operator, s, Echi, EL, mL, mChi, Lambda, T, mu_L, Fdeg_L, Fdeg_nu
            )
            assert np.isfinite(gamma).all()
            return gamma

        integrator = vegas.Integrator([[mL / (T + mL), 1.0], [-1.0, 1.0]])
        Gamma = integrator(f, **self.vegas_kwargs).mean
        return Gamma
