import sys
sys.path.append("src")
sys.path.append("tesseracts")

import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract
import wdm.tesseract_api as tesseract_api

jax.config.update("jax_enable_x64", True)

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

with Tesseract.from_image("tanh_sheath") as sheath_tx:
    with Tesseract.from_tesseract_api(tesseract_api) as tx:
        result = tx.apply(dict(
            Vc0=Vc0,
            Ip0=Ip0,
            a0=a0,
            N=N,
            Lp_prime=Lp_prime,
            Lz=Lz,
            R=R, L=L, C=C,
            sheath_tesseract_url=sheath_tx._client.url,
            dt=5e-8,
            t_end=1e-5,
        ))

