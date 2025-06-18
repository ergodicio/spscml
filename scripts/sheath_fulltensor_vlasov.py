import sys
sys.path.append("src")

import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

from spscml.fulltensor_vfp.solver import Solver
from spscml.plasma import TwoSpeciesPlasma
from spscml.grids import Grid, PhaseSpaceGrid
from spscml.utils import zeroth_moment, first_moment

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
            'val': 0.0
        },
    }
}

nu = 1.0
solver = Solver(plasma, 
                {'x': x_grid, 'electron': electron_grid, 'ion': ion_grid},
                flux_source_enabled=True,
                nu_ee=nu*5, nu_ii=nu)

solve = jax.jit(lambda: solver.solve(0.01, 1200, initial_conditions, boundary_conditions, 0.1))
result = solve()

E = solver.poisson_solve_from_fs(result, boundary_conditions)

fig, axes = plt.subplots(3, 1, figsize=(10, 8))

fe = result['electron']
fi = result['ion']

Se = jnp.linalg.svd(fe, compute_uv=False)
Si = jnp.linalg.svd(fi, compute_uv=False)

print("Se: ", Se / Se[0])
print("Si: ", Si / Si[0])

je = -1 * first_moment(fe, electron_grid)
ji = 1 * first_moment(fi, ion_grid)

axes[0].plot(-je, label='-je')
axes[0].plot(ji, label='ji')
axes[0].plot(je+ji, label='j')
axes[0].plot(E, label='E')
axes[0].legend()

ne = zeroth_moment(fe, electron_grid)
ni = zeroth_moment(fi, ion_grid)
#axes[1].plot(ne)
#axes[1].plot(ni)

axes[1].imshow(fe.T, origin='lower')
axes[2].imshow(fi.T, origin='lower')


plt.show()
