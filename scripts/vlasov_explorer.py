import sys
sys.path.append("src")
sys.path.append("tesseracts")

import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract
import wdm.tesseract_api as tesseract_api
import sheaths.vlasov.tesseract_api as sheath_tesseract_api
import jpu

jax.config.update("jax_enable_x64", True)

ureg = jpu.UnitRegistry()

R = 1.5e-3
L = 2.0e-7
C = 222*1e-6

# Charge to 50kV
Vc0 = 50*1e3

Lz = 0.5
Lp = -.4e-7
Lp_prime = Lp / Lz
L_tot = L - Lp

# Use the initial plasma from Fig 7 of Shumlak et al. (2012) as an example
a0 = 0.01
n0 = 6e22
N = n0 * (jnp.pi*a0**2)
Ip0 = -5e4

with Tesseract.from_tesseract_api(sheath_tesseract_api) as sheath_tx:
    Vp0 = 300 * (1 - 1e-6)
    N = 1e18
    Ip = sheath_tx.apply(dict(
        N=N, n=n0, T=200.0, Vp=Vp0, Lz=0.5
        ))["Ip"]
    print("Ip = ", Ip)
    j = Ip * (n0 / N)
    N_eq = (ureg.mu0 * (Ip*ureg.A)**2 / (2 * 8*jnp.pi * 20.0*ureg.eV)).to(ureg.m**-1).magnitude
    print("N_eq = ", N_eq)

    a0 = (N_eq / n0 / jnp.pi)**0.5

