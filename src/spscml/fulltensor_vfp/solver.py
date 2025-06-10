import jax.numpy as jnp
import jax
import equinox as eqx

from jaxtyping import PyTree
from typing import Callable
from functools import partial
from diffrax import diffeqsolve, Euler, Dopri5, ODETerm, PIDController, SaveAt

from ..plasma import TwoSpeciesPlasma
from ..grids import PhaseSpaceGrid
from ..rk import rk1, ssprk2
from ..muscl import slope_limited_flux_divergence
from ..poisson import poisson_solve
from ..utils import zeroth_moment, first_moment, second_moment
from .dougherty import lbo_operator_ij, species_info, lbo_operator_ij_L_diagonals

class Solver(eqx.Module):
    norm: dict
    plasma: TwoSpeciesPlasma
    grids: PyTree
    flux_source_enabled: bool
    nu_ee: float
    nu_ii: float

    """
    Solves the Vlasov-Fokker-Planck equation
    """
    def __init__(self,
                 plasma: TwoSpeciesPlasma, 
                 norm,
                 grids,
                 flux_source_enabled,
                 nu_ee, nu_ii):
        self.plasma = plasma
        self.norm = norm
        self.grids = grids
        self.flux_source_enabled = flux_source_enabled
        self.nu_ee = nu_ee
        self.nu_ii = nu_ii


    def step(self, fs, dt, bcs, f0):
        #return ssprk2(fs, lambda f: self.vlasov_fp_rhs(f, bcs, f0), dt)
        fs_prime = ssprk2(fs, lambda f: self.vlasov_fp_rhs(f, bcs, f0), dt)
        return self.implicit_collisions(fs_prime, dt)


    def solve(self, dt, Nt, initial_conditions, boundary_conditions, dtmax):
        f0 = {
            'electron': initial_conditions['electron'](*self.grids['electron'].xv),
            'ion': initial_conditions['ion'](*self.grids['ion'].xv),
        }

        t = 0.0
        f = f0

        '''
        vf = lambda t, f, args: self.vlasov_fp_rhs(f, *args)
        term = ODETerm(vf)
        solver = Dopri5()
        saveat = SaveAt(ts=[t_end])
        controller = PIDController(rtol=1e-5, atol=1e-5, dtmax=dtmax)

        sol = diffeqsolve(term, solver, y0=f0, args=(boundary_conditions, f0), dt0=dtmax/2, t0=0, t1=t_end)
        '''

        def onestep(i, f):
            return self.step(f, dt, boundary_conditions, f0)

        return jax.lax.fori_loop(0, Nt, onestep, f0)


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
            ion_source = total_ion_wall_flux * flux_source_weight * fi0
            ion_rhs = ion_rhs + total_ion_wall_flux * flux_source_weight * fi0

        assert electron_rhs.shape == fe.shape

        return dict(electron=electron_rhs, ion=ion_rhs)


    def implicit_collisions(self, fs, dt):
        fe = self.single_species_implicit_collisions(fs['electron'], 
                                                     self.grids['electron'], self.plasma.Ae, dt, self.nu_ee)
        fi = self.single_species_implicit_collisions(fs['ion'], 
                                                     self.grids['ion'], self.plasma.Ai, dt, self.nu_ii)
        return {'electron': fe, 'ion': fi}


    def single_species_implicit_collisions(self, f, grid, A, dt, nu):
        dl, d, du = lbo_operator_ij_L_diagonals(
                {"grid": grid, "n": 1.0,"A": A},
                {"T": 1.0, "u": 0.0, "lambda": nu},
                )

        nu = self.collision_frequency_shape_func().flatten()

        solve = lambda nu, f: jax.lax.linalg.tridiagonal_solve(-dt * nu*dl, 1 - dt * nu*d, -dt * nu*du, 
                                                               f[:, None]).flatten()
        f_next = jax.vmap(solve)(nu, f)
        return f_next



    def collision_frequency_shape_func(self):
        L = self.grids['x'].Lx

        midpt = L/3
        # Want 10 e-foldings between the midpoint (2/3rds of the way to the sheath)
        # and the wall
        efolding_dist = (L/6)/10
        x = self.grids['x'].xs
        h0 = lambda x: 1 + jnp.exp((x/efolding_dist) - midpt/efolding_dist)
        h = 1 / (0.5 * (h0(x) + h0(-x)))
        return jnp.expand_dims(h, axis=1)


    def vlasov_fp_single_species_rhs(self, f, E, A, Z, grid, bcs):
        # free streaming term
        f_bc_x = self.apply_bcs(f, bcs, 'x')

        v = jnp.expand_dims(grid.vs, axis=0)
        F = lambda left, right: jnp.where(v > 0, left * v, right * v)
        vdfdx = slope_limited_flux_divergence(f_bc_x, 'minmod', F, 
                                              grid.dx,
                                              axis=0)

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
