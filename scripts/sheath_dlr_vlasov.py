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
ion_grid = x_grid.extend_to_phase_space(6*vti, 150)
electron_grid = x_grid.extend_to_phase_space(6*vte, 150)

r = 20
def lowrank_factors(f, grid):
    X, S, V = jnp.linalg.svd(f, full_matrices=False)

    X = X.T[:r, :] / grid.dx**0.5
    S = jnp.diag(S[:r]) * (grid.dx * grid.dv)**0.5
    V = V[:r, :] / grid.dv**0.5

    assert X.shape == (r, 200)
    assert S.shape == (r, r)
    assert V.shape == (r, 150)

    return (X, S, V)

initial_conditions = { 
    'electron': lambda x, v: lowrank_factors(1 / (jnp.sqrt(2*jnp.pi)*vte) * jnp.exp(-Ae*(v**2) / (2*Te)), electron_grid),
    'ion': lambda x, v: lowrank_factors(1 / (jnp.sqrt(2*jnp.pi)*vti) * jnp.exp(-Ai*(v**2) / (2*Ti)), ion_grid)
}

solver = Solver(plasma, 
                r,
                {'x': x_grid, 'electron': electron_grid, 'ion': ion_grid},
                1.0, 1.0)

solve = jax.jit(lambda: solver.solve(0.001, 600, initial_conditions, 0.001))
result = solve()

Xt, S, V = result['electron']
fe = Xt.T @ S @ V

fig, axes = plt.subplots(3, 1, figsize=(10, 8))
axes[0].imshow(fe.T, origin='lower')
axes[0].set_aspect("auto")
axes[1].plot(V.T)
axes[2].plot(Xt.T)
plt.show()

