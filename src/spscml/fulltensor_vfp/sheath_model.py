import jax
import jax.numpy as jnp

from .solver import Solver
from ..utils import zeroth_moment, first_moment, temperature
from ..plasma import TwoSpeciesPlasma
from ..grids import Grid, PhaseSpaceGrid
from ..normalization import plasma_norm
from ..poisson import poisson_solve

import jpu


Lz_LAMBDA_D = 256


def make_plasma(norm):
    return TwoSpeciesPlasma(norm["omega_p_tau"], norm["omega_c_tau"], norm["nu_p_tau"], 
                            Ai=1.0, Ae=0.04, Zi=1.0, Ze=-1.0)


def reduced_mfp_for_sim(norm, Ae, Lz):
    interelectrode_gap = Lz * norm["ureg"].m
    mfp_fraction = ((0.75*Lz_LAMBDA_D * norm["lambda_D"]) / interelectrode_gap).to('').magnitude
    sim_mfp = mfp_fraction * (norm["lambda_mfp_spitzer"] / norm["lambda_D"]).to('').magnitude
    #sim_mfp = sim_mfp * (Ae * 1836)**0.5
    return sim_mfp
    return Lz_LAMBDA_D / 4


@jax.jit
def calculate_plasma_current(Vp, T, n, Lz, **kwargs):
    '''
    Calculates the plasma current carried by a plasma with the given temperature and
    number density across an electrode gap at the given voltage.

    params:
        - Vp: The electrode gap voltage [volts]
        - T: The plasma temperature [eV]
        - n: The plasma volumetric number density [m^-3]

    returns:
        - j: The space-averaged current density [amperes / m^2]
    '''
    norm = plasma_norm(T, n)
    ureg = norm["ureg"]

    Vp = (Vp * ureg.volt / norm["V0"]).magnitude
    plasma = make_plasma(norm)
    sim_mfp = reduced_mfp_for_sim(norm, plasma.Ae, Lz)

    Te = 1.0
    Ti = 1.0

    vte = jnp.sqrt(Te / plasma.Ae)
    vti = jnp.sqrt(Ti / plasma.Ai)

    Nx = Lz_LAMBDA_D*2
    x_grid = Grid(Nx, Lz_LAMBDA_D)
    ion_grid = x_grid.extend_to_phase_space(6*vti, 128)
    electron_grid = x_grid.extend_to_phase_space(6*vte, 128)

    initial_conditions = { 
        'electron': lambda x, v: 1 / (jnp.sqrt(2*jnp.pi)*vte) * jnp.exp(-plasma.Ae*(v**2) / (2*Te)),
        'ion': lambda x, v: 1 / (jnp.sqrt(2*jnp.pi)*vti) * jnp.exp(-plasma.Ai*(v**2) / (2*Ti))
    }

    left_ion_absorbing_wall = lambda f_in: jnp.where(ion_grid.vT > 0, 0.0, f_in)
    left_electron_absorbing_wall = lambda f_in: jnp.where(electron_grid.vT > 0, 0.0, f_in)
    right_ion_absorbing_wall = lambda f_in: jnp.where(ion_grid.vT < 0, 0.0, f_in)
    right_electron_absorbing_wall = lambda f_in: jnp.where(electron_grid.vT < 0, 0.0, f_in)

    boundary_conditions = {
        'electron': {
            'x': { 'left': left_electron_absorbing_wall, 'right': right_electron_absorbing_wall, },
            'v': { 'left': jnp.zeros_like, 'right': jnp.zeros_like, }
        },
        'ion': {
            'x': { 'left': left_ion_absorbing_wall, 'right': right_electron_absorbing_wall, },
            'v': { 'left': jnp.zeros_like, 'right': jnp.zeros_like, }
        },
        'phi': {
            'left': { 'type': 'Dirichlet', 'val': 0.0 },
            'right': { 'type': 'Dirichlet', 'val': Vp },
        }
    }

    nu_ee = vte / sim_mfp
    nu_ii = vti / sim_mfp

    solver = Solver(plasma, 
                    {'x': x_grid, 'electron': electron_grid, 'ion': ion_grid},
                    flux_source_enabled=True, nu_ee=nu_ee, nu_ii=nu_ii)

    s = 0.5
    dtmax = jnp.minimum(s * x_grid.dx / (6*vte), nu_ee / electron_grid.dv**2)

    solve = lambda: solver.solve(1 * dtmax, 5000, initial_conditions, boundary_conditions, dtmax)
    result = solve()
    je = -1 * first_moment(result['electron'], electron_grid)
    ji = 1 * first_moment(result['ion'], ion_grid)
    j = je + ji
    j_avg = jnp.sum(j) / x_grid.Nx

    ne = zeroth_moment(result['electron'], electron_grid)
    ni = zeroth_moment(result['ion'], ion_grid)
    Ti = temperature(result['ion'], plasma.Ai, ion_grid)

    j_avg = (j_avg * norm["j0"]).to(ureg.amperes / ureg.m**2).magnitude
    jax.debug.print("j = {}", j_avg)

    E = poisson_solve(x_grid, plasma, plasma.Zi*ni+plasma.Ze*ne, boundary_conditions)

    return dict(
            fe=result['electron'],
            fi=result['ion'],
            ion_grid=ion_grid,
            electron_grid=electron_grid,
            je=je,
            ji=ji,
            j_avg=j_avg,
            E=E,
            ni=ni,
            )
