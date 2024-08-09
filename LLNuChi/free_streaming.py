import numpy as np
import vegas

from utils.kinematics import get_s
from utils.thermodynamics import get_Fdeg
from utils.free_streaming import FreeStreamer
from LLNuChi.analytical import get_J


class FreeStreamer_LLNuChi(FreeStreamer):
    def get_dQdR(self, operator, R, **kwargs):
        # processes sorted by relevance
        dQdV_ann, dQdV_scat = 0.0, 0.0

        # L+ L- > nubar chibar
        dQdV_ann += self._get_dQdV_ann(
            operator=operator, antiparticle_finalstate=True, **kwargs
        )
        # L+ L- > nu chi
        dQdV_ann += self._get_dQdV_ann(
            operator=operator, antiparticle_finalstate=False, **kwargs
        )
        # L- nu > L- chi
        dQdV_scat += self._get_dQdV_scat(
            operator=operator, antilepton=False, antineutrino=False, **kwargs
        )
        # L- nubar > L- chibar
        dQdV_scat += self._get_dQdV_scat(
            operator=operator, antilepton=False, antineutrino=True, **kwargs
        )
        # L+ nu > L+ chi
        dQdV_scat += self._get_dQdV_scat(
            operator=operator, antilepton=True, antineutrino=False, **kwargs
        )
        # L+ nubar > L+ chibar
        dQdV_scat += self._get_dQdV_scat(
            operator=operator, antilepton=True, antineutrino=True, **kwargs
        )

        dQdV = dQdV_ann + dQdV_scat
        dQdR = 4 * np.pi * R**2 * dQdV
        return dQdR

    def _get_dQdV_ann(self, operator, antiparticle_finalstate, T, mu, model):
        mu_L = mu["L"]
        mu_nu = -mu["nu_L"] if antiparticle_finalstate else mu["nu_L"]
        mL, mChi, Lambda = model["mL"], model["mChi"], model["Lambda"]
        Fdeg_nu = get_Fdeg(0.0, T, mu_nu, is_boson=False)

        def get_integrand(s, E1, E2):
            J = get_J(
                "annihilation",
                operator,
                s=s,
                mL=mL,
                mChi=mChi,
                E1=E1,
                E2=E2,
                Fdeg=Fdeg_nu,
                Lambda=Lambda,
            )
            get_weighting = lambda E, mass, mu: (E**2 - mass**2) ** 0.5 / (
                np.exp((E - mu_L) / T) + 1
            )
            weighting = get_weighting(E1, mL, mu_L) * get_weighting(E2, mL, -mu_L)
            integrand = J * weighting
            mask1 = s > 4 * mL**2
            mask2 = s > mChi**2
            mask = np.logical_and(mask1, mask2)
            integrand[~mask] = 0.0
            return integrand

        @vegas.batchintegrand
        def f(x):
            a, b, costh = x[..., 0], x[..., 1], x[..., 2]
            E1, E2 = T * a / (1 - a), T * b / (1 - b)
            jac = T / (1 - a) ** 2 * T / (1 - b) ** 2
            s = get_s(mL, mL, E1, E2, costh)
            integrand = get_integrand(s, E1, E2) * jac
            return integrand

        integrator = vegas.Integrator(
            [[mL / (T + mL), 1.0], [mL / (T + mL), 1.0], [-1.0, 1.0]]
        )
        factor = integrator(f, **self.vegas_kwargs).mean
        dQdV = 1 / (32 * np.pi**4) * factor
        return dQdV

    def _get_dQdV_scat(self, operator, antilepton, antineutrino, T, mu, model):
        mu_L = -mu["L"] if antilepton else mu["L"]
        mu_nu = -mu["nu_L"] if antineutrino else mu["nu_L"]
        mL, mChi, Lambda = model["mL"], model["mChi"], model["Lambda"]
        Fdeg_L = get_Fdeg(mL, T, mu_L, is_boson=False)

        def get_integrand(s, EL, ENu):
            J = get_J(
                "scattering",
                operator,
                s=s,
                mL=mL,
                mChi=mChi,
                E1=EL,
                E2=ENu,
                Fdeg=Fdeg_L,
                Lambda=Lambda,
            )
            get_weighting = lambda E, mass, mu: (E**2 - mass**2) ** 0.5 / (
                np.exp((E - mu) / T) + 1
            )
            weighting = get_weighting(EL, mL, mu_L) * get_weighting(ENu, 0.0, mu_nu)
            integrand = J * weighting
            mask1 = s > mL**2
            mask2 = s > (mL + mChi) ** 2
            mask = np.logical_and(mask1, mask2)
            integrand[~mask] = 0.0
            return integrand

        @vegas.batchintegrand
        def f(x):
            a, b, costh = x[..., 0], x[..., 1], x[..., 2]
            EL, ENu = T * a / (1 - a), T * b / (1 - b)
            jac = T / (1 - a) ** 2 * T / (1 - b) ** 2
            s = get_s(mL, 0.0, EL, ENu, costh)
            integrand = get_integrand(s, EL, ENu) * jac
            return integrand

        integrator = vegas.Integrator(
            [[mL / (T + mL), 1.0], [mL / (T + mL), 1.0], [-1.0, 1.0]]
        )
        factor = integrator(f, **self.vegas_kwargs).mean
        dQdV = 1 / (32 * np.pi**4) * factor
        return dQdV
