import numpy as np


class FreeStreamer:
    def __init__(self):
        self.vegas_kwargs = {"nitn": 10, "neval": 1000, "alpha": 0.5}

    def get_Q(self, operator, sim_range, R, T, mu, **kwargs):
        n1, n2 = sim_range
        N = n2 - n1
        dQdRs = np.zeros(N)
        for i in range(N):
            mu_i = {
                particle: mu_particle[n1 + i] for particle, mu_particle in mu.items()
            }
            dQdRs[i] = self.get_dQdR(
                operator, R=R[n1 + i], T=T[n1 + i], mu=mu_i, **kwargs
            )
        Q = np.trapz(dQdRs, x=R[n1:n2])
        return Q

    def get_dQdR(self, operator, R, **kwargs):
        raise NotImplementedError
