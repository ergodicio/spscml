import sys
sys.path.append("src")

import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

from spscml.whole_device_model.solver import Solver
from spscml.plasma import TwoSpeciesPlasma

jax.config.update("jax_enable_x64", True)

Ae = 0.04
Ai = 1.0
plasma = TwoSpeciesPlasma(1.0, 1.0, 0.0, Ai, Ae, 1.0, -1.0)


def plasma_current(tx, Vp, N, T):
    Ip = apply_tesseract(tx, dict(Vp=jnp.array(Vp), 
                                  T=jnp.array(T), 
                                  N=jnp.array(N/1e18)))["Ip"]
    return dict(Ip=Ip*1e3)


def solver1():
    #Q0 = 200
    R = 1.0e-3
    L = 2.0e-7
    C = 222*1e-6

    # Charge to 25kV
    V0 = 50*1e3
    Q0 = C * V0

    Lp = -.4e-7
    L_tot = L - Lp
    alpha = (R) / (2*L_tot)
    omega_0 = 1 / jnp.sqrt(L_tot*C)
    jax.debug.print("alpha: {}", alpha)
    jax.debug.print("omega_0: {}", omega_0)
    zeta = alpha / omega_0 * (1+0j)
    jax.debug.print("zeta: {}", zeta)

    s1 = -omega_0 / (zeta + jnp.sqrt(zeta**2 - 1))
    s2 = -omega_0 * (zeta + jnp.sqrt(zeta**2 - 1))
    jax.debug.print("Eigenvals: {}, {}", s1, s2)

    Lz = 0.5
    n0 = 6e24
    a = 0.01
    N = n0 * jnp.pi * a**2

    solver = Solver(plasma, R, L, C, Lp, V0, Lz, N)
    ics = jnp.array([Q0, -5e4, 20.0, 6e22])

    dt = 1e-7

    Nt = 200
    with Tesseract.from_image("tanh_sheath") as tx:
        sheath_solver = lambda Vp, N, T: plasma_current(tx, Vp, N, T)
        carry, ys = solver.solve(dt, Nt, ics, sheath_solver)

    _, Vp, _ = carry
    jax.debug.print("final plasma voltage: {}", Vp)
    ts = jnp.linspace(0, (Nt-1)*dt, Nt)

    return ys, ts

def solver2():
    Q0 = 5.0
    R = 1.0
    L = 1.0
    C = 1.0

    V0 = Q0 / C

    Lp = -0.3
    L_tot = L - Lp
    alpha = R / (2*L_tot)
    omega_0 = 1 / jnp.sqrt(L_tot*C)
    jax.debug.print("alpha: {}", alpha)
    jax.debug.print("omega_0: {}", omega_0)

    solver = Solver(plasma, R, L, C, Lp, V0)

    ics = jnp.array([Q0, 0.0])
    dt = 0.1
    _, ys = solver.solve(dt, 200, ics, solver.estimate_plasma_current)

    return ys

ys, ts = solver1()

fig, axes = plt.subplots(4, 1, figsize=(14, 8))

axes[0].plot(ts, ys[:, 0], label='Capacitor charge [Coulombs]')
axes[1].plot(ts, ys[:, 1], label='Plasma current [Amperes]')
axes[2].plot(ts, ys[:, 2], label='Temperature')
axes[3].plot(ts, ys[:, 3], label='Density')

print(ys[-1, :])

plt.show()
