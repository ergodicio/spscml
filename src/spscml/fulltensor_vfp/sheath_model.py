import jax
import jax.numpy as jnp

from .solver import Solver
from ..utils import zeroth_moment, first_moment, temperature
from ..plasma import TwoSpeciesPlasma
from ..grids import Grid, PhaseSpaceGrid
from ..normalization import plasma_norm

import jpu

@jax.jit
def calculate_plasma_current(Vp, T, n, N):
    '''
    Calculates the plasma current carried by a plasma with the given temperature and
    number density across an electrode gap at the given voltage.

    params:
        - Vp: The electrode gap voltage [volts]
        - T: The plasma temperature [eV]
        - n: The plasma volumetric number density [m^-3]
        - N: The plasma linear number density [m^-1]

    returns:
        - Ip: The space-averaged current [amperes]
    '''
    Ae = 0.04
    Ai = 1.0

    norm = plasma_norm(T, n)
    ureg = norm["ureg"]

    Vp = (Vp * ureg.volt / norm["V0"]).magnitude

    Te = 1.0
    Ti = 1.0

    vte = jnp.sqrt(Te / Ae)
    vti = jnp.sqrt(Ti / Ai)

    omega_pe_tau = jnp.sqrt(1 / Ae) * norm["omega_p_tau"]
    jax.debug.print("omega pe: {}", omega_pe_tau)
    jax.debug.print("omega_c_tau: {}", norm["omega_c_tau"])
    jax.debug.print("nu_p_tau: {}", norm["nu_p_tau"])

    plasma = TwoSpeciesPlasma(norm["omega_p_tau"], norm["omega_c_tau"], norm["nu_p_tau"], 
                              Ai, Ae, 1.0, -1.0)

    jax.debug.print("lambda_D: {}", norm["lambda_D"])

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

    solver = Solver(plasma, 
                    norm,
                    {'x': x_grid, 'electron': electron_grid, 'ion': ion_grid},
                    flux_source_enabled=True)

    s = 0.5
    Nt = 8000
    solve = lambda: solver.solve(s / omega_pe_tau, Nt, initial_conditions, boundary_conditions)
    result = solve()
    je = -1 * first_moment(result['electron'], electron_grid)
    ji = 1 * first_moment(result['ion'], ion_grid)
    j = je + ji
    j_avg = jnp.sum(j) / x_grid.Nx

    ni = zeroth_moment(result['ion'], ion_grid)
    Ti = temperature(result['ion'], Ai, ion_grid)

    Ip = (j_avg * norm["j0"] * (N / n) * ureg.m**2).to(ureg.amperes)

    return dict(
            Ip=Ip.magnitude,
            fe=result['electron'],
            fi=result['ion'],
            ion_grid=ion_grid,
            electron_grid=electron_grid,
            je=je,
            ji=ji,
            ni=ni,
            )
