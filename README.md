# Supernova constraints

We use Supernova simulations to constrain BSM models with the Raffelt criterion. Energy loss rates in the free-streaming and trapping regimes are calculated following the formalism outlined in https://arxiv.org/abs/2307.03143. 

This repo aims to modularize the code used in https://arxiv.org/abs/2307.03143 (https://github.com/spinjo/SNforMuTau), and make it easier to adapt the code to new models.

## Constraints on the LLNuChi model

For this study, we extend the SM by a light dark-sector fermion $\chi$ that interacts with SM fermions through the effective interaction

$$\mathcal{L} \supset \frac{1}{\Lambda^2}(\bar \ell \Gamma_\ell \ell)(\bar\chi \Gamma_\chi \nu) + \mathrm{h.c.}.$$

We consider $\ell=e, \mu$ and different structures for the Dirac matrices $\Gamma_\ell, \Gamma_\chi$. Bounds on $\Lambda$ are evaluated as a function of $m_\chi$ for all operators. We compare the Supernova constraints with other constraints from astrophysics, cosmology and direct detection experiments. All results can be found in `LLNuChi/results`.