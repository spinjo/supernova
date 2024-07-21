import numpy as np
import utils.constants as c
from utils.supernova import get_mL_eff


class FreeStreamer:
    def __init__(self):
        self.vegas_kwargs = {"nitn": 10, "neval": 1000, "alpha": 0.5}

    def get_Q(self, operator, sim_range, R, T, mu, model, **kwargs):
        n1, n2 = sim_range
        N = n2 - n1
        dQdRs = np.zeros(N)
        mL = model["mL"]
        for i in range(N):
            if mL == c.me:
                # use electron effective mass
                model["mL"] = get_mL_eff(mL, T[n1 + i], mu["L"][n1 + i])
            mu_i = {
                particle: mu_particle[n1 + i] for particle, mu_particle in mu.items()
            }
            dQdRs[i] = self.get_dQdR(
                operator, R=R[n1 + i], T=T[n1 + i], mu=mu_i, model=model, **kwargs
            )
        Q = np.trapz(dQdRs, x=R[n1:n2])
        return Q

    def get_dQdR(self, operator, R, **kwargs):
        raise NotImplementedError
