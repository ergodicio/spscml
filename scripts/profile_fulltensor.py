import sys
sys.path.append("src")

import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

from spscml.fulltensor_vfp.solver import Solver
from spscml.plasma import TwoSpeciesPlasma
from spscml.grids import Grid, PhaseSpaceGrid
from spscml.utils import zeroth_moment, first_moment

from timeit import timeit

Te = 1.0
Ti = 1.0
ne = 1.0
Ae = 0.04
Ai = 1.0

vte = jnp.sqrt(Te / Ae)
vti = jnp.sqrt(Ti / Ai)

plasma = TwoSpeciesPlasma(1.0, 1.0, 0.0, Ai, Ae, 1.0, -1.0)

x_grid = Grid(400, 200)
ion_grid = x_grid.extend_to_phase_space(6*vti, 200)
electron_grid = x_grid.extend_to_phase_space(6*vte, 200)

initial_conditions = {
    'electron': lambda x, v: 1 / (jnp.sqrt(2*jnp.pi)*vte) * jnp.exp(-Ae*(v**2) / (2*Te)),
    'ion': lambda x, v: 1 / (jnp.sqrt(2*jnp.pi)*vti) * jnp.exp(-Ai*(v**2) / (2*Ti))
}

left_ion_absorbing_wall = lambda f_in: jnp.where(ion_grid.vT > 0, 0.0, f_in)
left_electron_absorbing_wall = lambda f_in: jnp.where(electron_grid.vT > 0, 0.0, f_in)
right_ion_absorbing_wall = lambda f_in: jnp.where(ion_grid.vT < 0, 0.0, f_in)
right_electron_absorbing_wall = lambda f_in: jnp.where(electron_grid.vT < 0, 0.0, f_in)

boundary_conditions = {
    'electron': {
        'x': {
            'left': left_electron_absorbing_wall,
            'right': right_electron_absorbing_wall,
        },
        'v': {
            'left': jnp.zeros_like,
            'right': jnp.zeros_like,
        }
    },
    'ion': {
        'x': {
            'left': left_ion_absorbing_wall,
            'right': right_electron_absorbing_wall,
        },
        'v': {
            'left': jnp.zeros_like,
            'right': jnp.zeros_like,
        }
    },
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
solver = Solver(plasma,
                {'x': x_grid, 'electron': electron_grid, 'ion': ion_grid},
                flux_source_enabled=True,
                nu_ee=nu*5, nu_ii=nu)

solve = jax.jit(lambda: solver.solve(0.01, 3000, initial_conditions, boundary_conditions, 0.1))
result = solve()

elapsed = timeit(lambda: jax.block_until_ready(solve()), number=1)
print(f"{elapsed} s")
