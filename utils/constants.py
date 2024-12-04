import math

# masses (all in MeV)
me = 0.51099895000
mmu = 105.6583755
mtau = 1776.86
mp = 938.272088
mn = 939.565420

# couplings
alphaEM = 1 / 137.035999084
e = (4 * math.pi * alphaEM) ** 0.5

# transforming units
erg2MeV = 6.2415e5
invs2MeV = 1 / 1.5e24 * 1e3
km2invMeV = 5.1e15
MeV2invfm = 5.1e-3
invcmtoMeV = 1 / 5.1e10

# supernova-specific constants
two_thirds = 2 / 3  # in case you dont trust astrophysics
