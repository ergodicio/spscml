import sys
sys.path.append("src")
sys.path.append("tesseracts")

import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract
import wdm.tesseract_api as tesseract_api
import sheaths.tanh_sheath.tesseract_api as sheath_tesseract_api
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

with Tesseract.from_image("vlasov_sheath") as sheath_tx:
    Vp0 = 300.0
    N = 1e18
    T0 = 20.0
    """
    Ip = sheath_tx.apply(dict(
        N=N, n=n0, T=T0, Vp=Vp0, Lz=0.5
        ))["Ip"]
    """
    Ip_prospective = -3593.5286319923143
    print("Ip_prospective = ", Ip_prospective)
    j = Ip_prospective * (n0 / N) * ureg.A * ureg.m**-2

    N_actual = ((8*jnp.pi * T0*ureg.eV * (n0*ureg.m**-3)**2) / (ureg.mu0 * j**2)).to(ureg.m**-1).magnitude
    print("N_actual = ", N_actual)

    Ip = (j * N_actual * ureg.m**-1 / (n0 * ureg.m**-3)).to(ureg.A).magnitude
    print("Ip actual = ", Ip)

    a0 = (N_actual / n0 / jnp.pi)**0.5

    with Tesseract.from_tesseract_api(tesseract_api) as tx:
        result = tx.apply(dict(
            Vc0=Vc0,
            Ip0=Ip,
            a0=a0,
            N=N_actual,
            Lp_prime=Lp_prime,
            Lz=Lz,
            R=R, L=L, C=C,
            sheath_tesseract_url=sheath_tx._client.url,
            dt=5e-8,
            t_end=2e-7,
        ))

