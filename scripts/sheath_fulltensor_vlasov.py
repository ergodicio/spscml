import sys
sys.path.append("src")

import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

import jpu

from spscml.poisson import poisson_solve
from spscml.fulltensor_vlasov.solver import Solver
from spscml.plasma import TwoSpeciesPlasma
from spscml.grids import Grid, PhaseSpaceGrid
from spscml.utils import zeroth_moment, first_moment
from spscml.normalization import plasma_norm


Lz_LAMBDA_D = 256

def reduced_mfp_for_sim(norm, Ae, Lz):
    interelectrode_gap = Lz * norm["ureg"].m
    mfp_fraction = ((0.75*Lz_LAMBDA_D * norm["lambda_D"]) / interelectrode_gap).to('').magnitude
    sim_mfp = mfp_fraction * (norm["lambda_mfp_spitzer"] / norm["lambda_D"]).to('').magnitude
    #sim_mfp = sim_mfp * (Ae * 1836)**0.5
    return sim_mfp
    return Lz_LAMBDA_D / 4


ureg = jpu.UnitRegistry()

Te = 1.0
Ti = 1.0
ne = 1.0
Ae = 0.04
Ai = 1.0

vte = jnp.sqrt(Te / Ae)
vti = jnp.sqrt(Ti / Ai)




norm = plasma_norm(20, 6e22)

sim_mfp = reduced_mfp_for_sim(norm, Ae, 0.5)

plasma = TwoSpeciesPlasma(norm["omega_p_tau"], norm["omega_c_tau"], norm["nu_p_tau"],
                        Ai=1.0, Ae=0.04, Zi=1.0, Ze=-1.0)


#plasma = TwoSpeciesPlasma(1.0, 2e-4, 0.0, Ai, Ae, 1.0, -1.0)

x_grid = Grid(512, 256)
ion_grid = x_grid.extend_to_phase_space(6*vti, 128)
electron_grid = x_grid.extend_to_phase_space(6*vte, 128)

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
            'val': 1500/(norm["V0"].magnitude)
        },
    }
}

nu = 1.0
solver = Solver(plasma,
                {'x': x_grid, 'electron': electron_grid, 'ion': ion_grid},
                flux_source_enabled=True,
                nu_ee=vte / sim_mfp, nu_ii=vti / sim_mfp)

CFL = 0.5
dt = CFL * x_grid.dx / (6*electron_grid.vmax)

solve = jax.jit(lambda: solver.solve(dt, 10000, initial_conditions, boundary_conditions, 0.1))
result = solve()

fe = result['electron']
fi = result['ion']

Se = jnp.linalg.svd(fe, compute_uv=False)
Si = jnp.linalg.svd(fi, compute_uv=False)

print("Se: ", Se / Se[0])
print("Si: ", Si / Si[0])

je = -1 * first_moment(fe, electron_grid)
ji = 1 * first_moment(fi, ion_grid)

j = je + ji
j_avg = jnp.sum(j) / x_grid.Nx

j_avg = (j_avg * norm["j0"]).to(ureg.amperes / ureg.m**2).magnitude
jax.debug.print("")
jax.debug.print("Result of the simulation:")
jax.debug.print("    Average current density j = {}, [amperes / meters^2]", j_avg)
jax.debug.print("")
jax.debug.print("")



ne = zeroth_moment(fe, electron_grid)
ni = zeroth_moment(fi, ion_grid)

E = poisson_solve(x_grid, plasma, ni-ne, boundary_conditions)

fig, axes = plt.subplots(4, 1, figsize=(10, 8))
axes[0].imshow(fe.T, origin='lower')
axes[0].set_aspect("auto")
axes[0].set_title("$f_e$")
axes[1].imshow(fi.T, origin='lower')
axes[1].set_aspect("auto")
axes[1].set_title("$f_i$")
axes[2].plot(ji.T, label='ji')
axes[2].plot(-je.T, label='-je')
axes[2].plot((ji+je).T, label='j')
axes[2].plot(E, label='E')
axes[2].legend()
axes[3].plot(ne, label='ne')
axes[3].plot(ni, label='ni')
axes[3].legend()
plt.show()
