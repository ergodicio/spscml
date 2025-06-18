import sys
sys.path.append("src")

import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

from spscml.straightforward_dlra.solver import Solver
from spscml.plasma import TwoSpeciesPlasma
from spscml.grids import Grid, PhaseSpaceGrid
from spscml.utils import zeroth_moment, first_moment

from timeit import timeit

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", False)

Te = 1.0
Ti = 1.0
ne = 1.0
Ae = 0.04
Ai = 1.0

vte = jnp.sqrt(Te / Ae)
vti = jnp.sqrt(Ti / Ai)

plasma = TwoSpeciesPlasma(1.0, 1.0, 0.0, Ai, Ae, 1.0, -1.0)

x_grid = Grid(200, 100)
ion_grid = x_grid.extend_to_phase_space(8*vti, 100)
electron_grid = x_grid.extend_to_phase_space(8*vte, 100)
grids = {'x': x_grid, 'electron': electron_grid, 'ion': ion_grid}

r = 10
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
            'val': 4.3
        },
    }
}

nu = 1.0
solver = Solver(plasma, r, grids, nu*5, nu)

dtmax = x_grid.dx / electron_grid.vmax / 10
print("dt = ", dtmax)

solve = jax.jit(lambda: solver.solve(0.01/2, 3000, initial_conditions, boundary_conditions, 0.01))
# Warm up the jit
solve()
elapsed = timeit(solve, number=1)
print(f"{elapsed} s")

#with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    #jax.block_until_ready(solve())

