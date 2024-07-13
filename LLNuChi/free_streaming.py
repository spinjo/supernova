import numpy as np
import vegas

from utils.kinematics import get_s
from utils.thermodynamics import get_Fdeg
from utils.free_streaming import FreeStreaming
from LLNuChi.analytical import get_J

"""
TODO
- Also integrate over R within this class?
- Could parallelize over simulation points...
"""


class FreeStreaming_LLNuChi(FreeStreaming):
    def __init__(self):
        # vegas parameters
        # verbose / logger
        self.nitn = 10
        self.neval = 1000
        self.alpha = 0.5

    def get_dQdR(self, operator, R, **kwargs):
        # processes sorted by relevance
        dQdV_ann, dQdV_scat = 0.0, 0.0

        # L+ L- > nubar chibar
        dQdV_ann += self.get_dQdV_ann(
            operator=operator, antiparticle_finalstate=True, **kwargs
        )
        # L+ L- > nu chi
        dQdV_ann += self.get_dQdV_ann(
            operator=operator, antiparticle_finalstate=False, **kwargs
        )
        # L- nu > L- chi
        dQdV_scat += self.get_dQdV_scat(
            operator=operator, antilepton=False, antineutrino=False, **kwargs
        )
        # L- nubar > L- chibar
        dQdV_scat += self.get_dQdV_scat(
            operator=operator, antilepton=False, antineutrino=True, **kwargs
        )
        # L+ nu > L+ chi
        dQdV_scat += self.get_dQdV_scat(
            operator=operator, antilepton=True, antineutrino=False, **kwargs
        )
        # L+ nubar > L+ chibar
        dQdV_scat += self.get_dQdV_scat(
            operator=operator, antilepton=True, antineutrino=True, **kwargs
        )

        dQdV = dQdV_ann + dQdV_scat
        dQdR = 4 * np.pi * R**2 * dQdV
        return dQdR

    def get_dQdV_ann(self, operator, antiparticle_finalstate, T, mu, model):
        mu_L = mu["e"]
        mu_nu = -mu["nu_e"] if antiparticle_finalstate else mu["nu_e"]
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
            assert np.isfinite(integrand).all()
            return integrand

        integrator = vegas.Integrator(
            [[mL / (T + mL), 1.0], [mL / (T + mL), 1.0], [-1.0, 1.0]]
        )
        factor = integrator(f, nitn=self.nitn, neval=self.neval, alpha=self.alpha).mean
        dQdV = 1 / (32 * np.pi**4) * factor
        return dQdV

    def get_dQdV_scat(self, operator, antilepton, antineutrino, T, mu, model):
        mu_L = -mu["e"] if antilepton else mu["e"]
        mu_nu = -mu["nu_e"] if antineutrino else mu["nu_e"]
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
            assert np.isfinite(integrand).all()
            return integrand

        integrator = vegas.Integrator(
            [[mL / (T + mL), 1.0], [mL / (T + mL), 1.0], [-1.0, 1.0]]
        )
        factor = integrator(f, nitn=self.nitn, neval=self.neval, alpha=self.alpha).mean
        dQdV = 1 / (32 * np.pi**4) * factor
        return dQdV
