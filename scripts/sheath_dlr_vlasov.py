import sys
sys.path.append("src")

import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

from spscml.straightforward_dlra.solver import Solver
from spscml.plasma import TwoSpeciesPlasma
from spscml.grids import Grid, PhaseSpaceGrid
from spscml.utils import zeroth_moment, first_moment

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)

Te = 1.0
Ti = 1.0
ne = 1.0
Ae = 0.04
Ai = 1.0

vte = jnp.sqrt(Te / Ae)
vti = jnp.sqrt(Ti / Ai)

plasma = TwoSpeciesPlasma(1.0, 1.0, 0.0, Ai, Ae, 1.0, -1.0)

x_grid = Grid(200, 100)
ion_grid = x_grid.extend_to_phase_space(6*vti, 50)
electron_grid = x_grid.extend_to_phase_space(6*vte, 50)
grids = {'x': x_grid, 'electron': electron_grid, 'ion': ion_grid}

r = 16
def lowrank_factors(f, grid):
    X, S, V = jnp.linalg.svd(f, full_matrices=False)

    X = X.T[:r, :] / grid.dx**0.5
    S = jnp.diag(S[:r]) * (grid.dx * grid.dv)**0.5
    V = V[:r, :] / grid.dv**0.5

    return (X, S, V)

initial_conditions = { 
    'electron': lambda x, v: lowrank_factors(1 / (jnp.sqrt(2*jnp.pi)*vte) * jnp.exp(-Ae*(v**2) / (2*Te)), electron_grid),
    'ion': lambda x, v: lowrank_factors(1 / (jnp.sqrt(2*jnp.pi)*vti) * jnp.exp(-Ai*(v**2) / (2*Ti)), ion_grid)
}
boundary_conditions = {
    'phi': {
        'left': {
            'type': 'Dirichlet',
            'val': 0.0
        },
        'right': {
            'type': 'Dirichlet',
            'val': 0.0
        },
    }
}

solver = Solver(plasma, r, grids, 1.0, 1.0)

dv = ion_grid.dv
max_dt = 0.3 * dv**2
print("max_dt = ", max_dt)

solve = jax.jit(lambda: solver.solve(0.01 / 10, 40, initial_conditions, boundary_conditions, 0.001))
result = solve()

Xt, S, V = result['electron']
fe = Xt.T @ S @ V
Xt, S, V = result['ion']
fi = Xt.T @ S @ V

E = solver.solve_poisson(result, grids, boundary_conditions)

fig, axes = plt.subplots(3, 1, figsize=(10, 8))
axes[0].imshow(fe.T, origin='lower')
axes[0].set_aspect("auto")
axes[1].imshow(fi.T, origin='lower')
axes[1].set_aspect("auto")
axes[2].plot(E)
plt.show()

