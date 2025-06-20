import equinox as eqx
import sys
sys.path.append("src")
sys.path.append("tesseracts")

import os

import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract
import wdm.tesseract_api as tesseract_api
import sheaths.tanh_sheath.tesseract_api as sheath_tesseract_api
import jpu

# Data gathering loop

ns = 10 ** jnp.linspace(20, 25, 10)
Ts = 10 ** jnp.linspace(1, 4, 10)
Vs = 10 ** jnp.linspace(1, 4, 10)

Lzs, ns, Ts, Vs = jnp.meshgrid(Lzs, ns, Ts, Vs)

with Tesseract.from_image("tanh_sheath") as tx:
    for params in zip(Lzs.flatten(), ns.flatten(), Ts.flatten(), Vs.flatten()):
        Lz, n, T, V = params
        j = ts.apply(dict(n=n, T=T, Vp=V, Lz=Lz))["j"]



