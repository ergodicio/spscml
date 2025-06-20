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
import optimistix as optx

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
n0 = 6e22 * ureg.m**-3
Ip_target = -100.0*ureg.kA
T0 = 20.0 * ureg.eV

Z = 1.0

def find_initial_voltage(sheath_tx, Ip_target):
    jax.debug.print("Searching for voltage that will drive {} through the plasma.", Ip_target)

    Vp_guess = 500.0 # [volts]

    def residual(guess, args):
        j = apply_tesseract(sheath_tx, dict(
            n=jnp.array(n0.magnitude), T=jnp.array(T0.magnitude), Vp=jnp.array(guess), Lz=jnp.array(0.5)
            ))["j"] * (ureg.A / ureg.m**2)
        N = ((8*jnp.pi * (1 + Z) * T0 * n0**2) / (ureg.mu0 * j**2)).to(ureg.m**-1)
        Ip = (j * N / n0).to(ureg.A)
        jax.debug.print("Ip was {}", Ip)
        return (Ip-Ip_target).magnitude

    solver = optx.Newton(rtol=1e-8, atol=1e-8)
    root = optx.root_find(residual, solver, Vp_guess, max_steps=5, throw=False,
                          options={'lower': 1.0}).value

    return root


# Try out grad of initial voltage first

with Tesseract.from_image("vlasov_sheath") as sheath_tx:
    Vp0 = find_initial_voltage(sheath_tx, Ip_target) * ureg.volts
    jax.debug.print("Initial voltage: {}", Vp0)

    f = lambda Ip: find_initial_voltage(sheath_tx, Ip)
    g = jax.grad(f)(Ip_target)
    jax.debug.print("grad: {}", g)


sys.exit(0)


with Tesseract.from_image("tanh_sheath") as sheath_tx:
    Vp0 = find_initial_voltage(sheath_tx) * ureg.volts

    j = sheath_tx.apply(dict(
        n=n0.magnitude, T=T0.magnitude, Vp=Vp0.magnitude, Lz=0.5
        ))["j"] * (ureg.A / ureg.m**2)
    N = ((8*jnp.pi * (1 + Z) * T0 * n0**2) / (ureg.mu0 * j**2)).to(ureg.m**-1)
    Ip = (j * N / n0).to(ureg.A)
    a0 = ((N / n0 / jnp.pi)**0.5).to(ureg.m)

    with Tesseract.from_tesseract_api(tesseract_api) as tx:
        result = tx.apply(dict(
            Vc0=Vc0,
            Ip0=Ip.magnitude,
            a0=a0.magnitude,
            N=N.magnitude,
            Lp_prime=Lp_prime,
            Lz=Lz,
            R=R, L=L, C=C,
            sheath_tesseract_url=sheath_tx._client.url,
            dt=5e-8,
            t_end=1e-5,
        ))

