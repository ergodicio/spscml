import jax.numpy as jnp
import jax
import equinox as eqx

from jaxtyping import PyTree
from typing import Callable

from ..plasma import TwoSpeciesPlasma
from ..grids import PhaseSpaceGrid
from ..rk import ssprk2
from ..muscl import slope_limited_flux_divergence
from ..poisson import poisson_solve
from ..utils import zeroth_moment, first_moment

class Solver(eqx.Module):
    plasma: TwoSpeciesPlasma
    grids: PyTree = eqx.field(static=True)
    flux_source_enabled: bool

    """
    Solves the Vlasov-Fokker-Planck equation
    """
    def __init__(self,
                 plasma: TwoSpeciesPlasma, 
                 grids,
                 flux_source_enabled):
        self.plasma = plasma
        self.grids = grids
        self.flux_source_enabled = flux_source_enabled


    def max_dt(self):
        CFL = 0.8
        grid = self.grids['electron']
        free_streaming_limit = CFL * grid.dx / grid.vmax
        omega_pe = self.plasma.omega_p_tau / jnp.sqrt(self.plasma.Ae)
        omega_pe_limit = 0.2 / omega_pe
        return jnp.minimum(omega_pe_limit, free_streaming_limit)


    def step(self, fs, dt, bcs, f0):
        return ssprk2(fs, lambda f: self.vlasov_fp_rhs(f, bcs, f0), dt)


    def solve(self, dt, Nt, initial_conditions, boundary_conditions):
        f0 = {
            'electron': initial_conditions['electron'](*self.grids['electron'].xv),
            'ion': initial_conditions['ion'](*self.grids['ion'].xv),
        }

        t = 0.0
        dt = self.max_dt()
        f = f0

        def one_step(i, f):
            return self.step(f, dt, boundary_conditions, f0)

        return jax.lax.fori_loop(0, Nt, one_step, f0)


    def n(self, f, grid):
        return jnp.sum(f, axis=1) * grid.dv


    def rho_c(self, fs):
        fe = fs['electron']
        fi = fs['ion']
        rho_c = self.plasma.Ze * self.n(fe, self.grids['electron']) + \
                self.plasma.Zi * self.n(fi, self.grids['ion'])
        return rho_c


    def poisson_solve_from_fs(self, fs, boundary_conditions):
        E = poisson_solve(self.grids['x'], self.plasma, self.rho_c(fs), boundary_conditions)
        return E


    def vlasov_fp_rhs(self, fs, boundary_conditions, f0):
        fe = fs['electron']
        fi = fs['ion']
        E = self.poisson_solve_from_fs(fs, boundary_conditions)
        
        electron_rhs = self.vlasov_fp_single_species_rhs(fe, E, self.plasma.Ae, self.plasma.Ze, 
                                                         self.grids['electron'],
                                                         boundary_conditions['electron'])
        ion_rhs = self.vlasov_fp_single_species_rhs(fi, E, self.plasma.Ai, self.plasma.Zi, 
                                                         self.grids['ion'],
                                                         boundary_conditions['ion'])

        if self.flux_source_enabled:
            ion_particle_flux = first_moment(fi, self.grids['ion'])
            total_ion_wall_flux = -ion_particle_flux[0] + ion_particle_flux[-1]

            grid = self.grids['x']
            L_flux_source = grid.Lx / 4
            flux_source_weight = jnp.where(jnp.abs(grid.xs) < L_flux_source, 
                                           1 / L_flux_source - jnp.abs(grid.xs) / L_flux_source**2,
                                           0.0)
            flux_source_weight = jnp.expand_dims(flux_source_weight, axis=1)
            fe0 = jnp.expand_dims(f0['electron'][self.grids['x'].Nx // 2, :], axis=0)
            fi0 = jnp.expand_dims(f0['ion'][self.grids['x'].Nx // 2, :], axis=0)

            electron_rhs = electron_rhs + total_ion_wall_flux * flux_source_weight * fe0
            ion_rhs = ion_rhs + total_ion_wall_flux * flux_source_weight * fi0

        # Implement dumb electron drag term
        Ne = jnp.expand_dims(zeroth_moment(fe, self.grids['electron']), axis=1)
        ve = self.grids['electron'].vT
        Ae = self.plasma.Ae
        electron_stationary_maxwellian = Ne / jnp.sqrt(2*jnp.pi * 1 / Ae) * jnp.exp(-Ae*(ve**2) / (2))
        nu_ei = 0.1
        electron_rhs = electron_rhs + nu_ei * (electron_stationary_maxwellian - fe)

        # TODO: implement cross-species collision term

        return dict(electron=electron_rhs, ion=ion_rhs)


    def vlasov_fp_single_species_rhs(self, f, E, A, Z, grid, bcs):
        # free streaming term
        f_bc_x = self.apply_bcs(f, bcs, 'x')

        v = jnp.expand_dims(grid.vs, axis=0)
        F = lambda left, right: jnp.where(v > 0, left * v, right * v)
        vdfdx = slope_limited_flux_divergence(f_bc_x, 'minmod', F, grid.dx, axis=0)

        # electrostatic acceleration term
        f_bc_v = self.apply_bcs(f, bcs, 'v')

        E = jnp.expand_dims(E, axis=1)
        fac = self.plasma.omega_c_tau * Z / A
        F = lambda left, right: jnp.where(fac * E > 0, left * fac * E, right * fac * E)
        Edfdv = slope_limited_flux_divergence(f_bc_v, 'minmod', F, grid.dv, axis=1)

        # TODO: implement Fokker-Planck operator

        return -vdfdx - Edfdv


    def apply_bcs(self, f, bcs, dim):
        bc = bcs[dim]
        if dim == 'x':
            axis = 0
        elif dim == 'v':
            axis = 1

        if axis == 0:
            if bc == 'periodic':
                left = f[-2:, :]
                right = f[:2, :]
            else:
                left = bc['left'](f[0:2, :])
                right = bc['right'](f[-2:, :])
        elif axis == 1:
            if bc == 'periodic':
                left = f[:, -2:]
                right = r[:, :2]
            else:
                left = bc['left'](f[:, 0:2])
                right = bc['right'](f[:, -2:])

        return jnp.concatenate([left, f, right], axis=axis)
