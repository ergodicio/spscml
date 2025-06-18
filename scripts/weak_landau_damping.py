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
Ae = 1.0
Ai = 1.0

vte = jnp.sqrt(Te / Ae)
vti = jnp.sqrt(Ti / Ai)

plasma = TwoSpeciesPlasma(1.0, 1.0, 0.0, Ai, Ae, 0.0, 1.0)

x_grid = Grid(200, 4*jnp.pi)
ion_grid = x_grid.extend_to_phase_space(6*vti, 200)
electron_grid = x_grid.extend_to_phase_space(6*vte, 200)
grids = {'x': x_grid, 'electron': electron_grid, 'ion': ion_grid}

r = 8
def lowrank_factors(f, grid):
    X, S, V = jnp.linalg.svd(f, full_matrices=False)

    X = X.T[:r, :] / grid.dx**0.5
    S = jnp.diag(S[:r]) * (grid.dx * grid.dv)**0.5
    V = V[:r, :] / grid.dv**0.5

    return (X, S, V)

ne = 1 + 0.01 * jnp.cos(electron_grid.xs / 2)
initial_conditions = { 
                      'electron': lambda x, v: lowrank_factors(ne[:, None] / (jnp.sqrt(2*jnp.pi)*vte) * jnp.exp(-Ae*(v**2) / (2*Te)), electron_grid),
    'ion': lambda x, v: lowrank_factors(1 / (jnp.sqrt(2*jnp.pi)*vti) * jnp.exp(-Ai*(v**2) / (2*Ti)), ion_grid)
}
boundary_conditions = {
    'phi': 'periodic'
}

nu = 0.0
solver = Solver(plasma, r, grids, nu*5, nu, boundary_type='Periodic')

dtmax = x_grid.dx / electron_grid.vmax / 10
print("dt = ", dtmax)

solve = jax.jit(lambda: solver.solve(0.004, 4000, initial_conditions, boundary_conditions, dtmax))
solution = solve()

frame = lambda i: jax.tree.map(lambda ys: ys[i, ...], solution.ys)
Es = [solver.solve_poisson(frame(i), grids, boundary_conditions) for i in range(100)]

E2s = jnp.array([jnp.sum(E**2/2) * x_grid.dx for E in Es])


Xt, S, V = solution.ys['electron']
fe0 = Xt[0, ...].T @ S[0, ...] @ V[0, ...]
fe = Xt[-1, ...].T @ S[-1, ...] @ V[-1, ...]

fig, axes = plt.subplots(2, 1, figsize=(10, 8))
axes[0].imshow((fe - fe0).T, origin='lower')
axes[0].set_aspect('auto')
axes[1].plot(solution.ts, jnp.log10(E2s))

plt.show()
