import sys
sys.path.append("src")

import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

from spscml.linear_density.solver import Solver
from spscml.plasma import TwoSpeciesPlasma
from spscml.grids import Grid


Ae = 0.04
Ai = 1.0

plasma = TwoSpeciesPlasma(1.0, 1.0, 0.0, Ai, Ae, 1.0, -1.0)
x_grid = Grid(100, 100)

initial_conditions = {
    'electron': { 'N': jnp.ones_like, 'Nuz': jnp.zeros_like, },
    'ion': { 'N': jnp.ones_like, 'Nuz': jnp.zeros_like, },
}

boundary_conditions = { 'phi': {'left': 0.0, 'right': 1.0} }

solver = Solver(plasma, x_grid)

q = solver.solve(0.1, initial_conditions, boundary_conditions)

plt.plot(x_grid.xs, q['electron']['N'])
plt.plot(x_grid.xs, q['ion']['N'])

plt.show()
