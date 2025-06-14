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
import sheaths.tanh_sheath.tesseract_api as tanh_sheath_tesseract_api
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

Z = 1.0

# Use the initial plasma from Fig 7 of Shumlak et al. (2012) as an example
with Tesseract.from_tesseract_api(sheath_tesseract_api) as sheath_tx:
    n0 = 6e22 * ureg.m**-3
    T0 = 20.0 * ureg.eV

    # With 1.5kV, we get approximately 50kA of total current.
    Vp0 = 1500.0 * ureg.volts

    j = sheath_tx.apply(dict(
        n=n0.magnitude, T=T0.magnitude, Vp=Vp0.magnitude, Lz=0.5
        ))["j"] * (ureg.A / ureg.m**2)
    print("j  = ", j.to(ureg.MA / ureg.m**2))

    N = ((8*jnp.pi * (1 + Z) * T0 * n0**2) / (ureg.mu0 * j**2)).to(ureg.m**-1)
    print("N = ", N)

    Ip = (j * N / n0).to(ureg.A)
    print("Ip = ", Ip)

