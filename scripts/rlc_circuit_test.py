import sys
sys.path.append("src")

import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

from spscml.rlc_circuit.solver import Solver
from spscml.plasma import TwoSpeciesPlasma

jax.config.update("jax_enable_x64", True)

Ae = 0.04
Ai = 1.0
plasma = TwoSpeciesPlasma(1.0, 1.0, 0.0, Ai, Ae, 1.0, -1.0)

Q0 = 5.0
R = 0.3
L = .5
C = 1.0

V0 = Q0 / C

solver = Solver(plasma, R, L, C, -0.5, V0)

ics = jnp.array([Q0, 0.0])
_, ys = solver.solve(0.01, 160, ics)

plt.plot(ys)
plt.show()
