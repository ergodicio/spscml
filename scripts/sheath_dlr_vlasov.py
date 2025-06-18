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

r = 20
def lowrank_factors(f, grid):
    X, S, V = jnp.linalg.svd(f, full_matrices=False)

    X = X.T[:r, :] / grid.dx**0.5
    S = jnp.diag(S[:r]) * (grid.dx * grid.dv)**0.5
    V = V[:r, :] / grid.dv**0.5

    return (X, S, V)

#ne = 1 + 0.1 * jnp.cos(2*jnp.pi * electron_grid.xs / 100 * 6)
ne = jnp.ones(x_grid.Nx)
initial_conditions = { 
                      'electron': lambda x, v: lowrank_factors(ne[:, None] / (jnp.sqrt(2*jnp.pi)*vte) * jnp.exp(-Ae*(v**2) / (2*Te)), electron_grid),
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

nu = 1.0
solver = Solver(plasma, r, grids, nu*5, nu)

dtmax = x_grid.dx / electron_grid.vmax / 10
print("dt = ", dtmax)

solve = jax.jit(lambda: solver.solve(0.01, 3000, initial_conditions, boundary_conditions, 0.01))
result = solve()

Xt, S, V = result['electron']
fe = Xt.T @ S @ V
ne = Xt.T @ S @ (V @ jnp.ones(electron_grid.Nv)) * electron_grid.dv
je = -1 * Xt.T @ S @ (V @ electron_grid.vs) * electron_grid.dv
print(je.shape)
assert je.shape == (electron_grid.Nx,)

Xt, S, V = result['ion']
fi = Xt.T @ S @ V
ni = Xt.T @ S @ (V @ jnp.ones(ion_grid.Nv)) * ion_grid.dv
ji = Xt.T @ S @ (V @ ion_grid.vs) * ion_grid.dv

E = solver.solve_poisson_ys(result, grids, boundary_conditions)


fig, axes = plt.subplots(4, 1, figsize=(10, 8))
axes[0].imshow(fe.T, origin='lower')
axes[0].set_aspect("auto")
axes[1].imshow(fi.T, origin='lower')
axes[1].set_aspect("auto")
#axes[2].plot(E)
axes[3].plot(result['electron'][0][:4, :].T)
axes[2].plot(ji.T)
axes[2].plot((ji+je).T)
axes[2].plot(E, label='E')
axes[2].legend()
plt.show()

