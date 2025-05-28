import jax.numpy as jnp
import jax
import equinox as eqx

from jaxtyping import PyTree
from typing import Callable

from ..plasma import TwoSpeciesPlasma
from ..grids import PhaseSpaceGrid
from ..rk import ssprk2
from ..muscl import slope_limited_flux_divergence

class Solver(eqx.Module):
    plasma: TwoSpeciesPlasma
    grids: PyTree = eqx.field(static=True)

    """
    Solves the Vlasov-Fokker-Planck equation
    """
    def __init__(self,
                 plasma: TwoSpeciesPlasma, 
                 grids):
        self.plasma = plasma
        self.grids = grids


    def max_dt(self):
        CFL = 0.8
        grid = self.grids['electron']
        free_streaming_limit = CFL * grid.dx / grid.vmax
        omega_pe = self.plasma.omega_p_tau / jnp.sqrt(self.plasma.Ae)
        omega_pe_limit = 0.2 / omega_pe
        return jnp.minimum(omega_pe_limit, free_streaming_limit)


    def step(self, fs, dt, bcs):
        return ssprk2(fs, lambda f: self.vlasov_fp_rhs(f, bcs), dt)


    def solve(self, dt, Nt, initial_conditions, boundary_conditions):
        f0 = {
            'electron': initial_conditions['electron'](*self.grids['electron'].xv),
            'ion': initial_conditions['ion'](*self.grids['ion'].xv),
        }

        t = 0.0
        dt = self.max_dt()
        f = f0

        def one_step(i, f):
            return self.step(f, dt, boundary_conditions)

        return jax.lax.fori_loop(0, Nt, one_step, f0)


    def n(self, f, grid):
        return jnp.sum(f, axis=1) * grid.dv


    def poisson_solve(self, rho_c, boundary_conditions):
        """
        Solves the Poisson equation nabla^2 phi = -omega_c_tau / omega_p_tau^2 * rho_c.

        Returns E = -nabla phi
        """
        oct = self.plasma.omega_c_tau
        opt = self.plasma.omega_p_tau
        rhs = -oct / opt**2 * rho_c

        grid = self.grids['x']
        
        left_bc = boundary_conditions['phi']['left']
        right_bc = boundary_conditions['phi']['right']
        left_bc_type = left_bc['type']
        right_bc_type = right_bc['type']

        dx = grid.dx

        # Apply phi boundary conditions
        if left_bc_type == 'Dirichlet':
            rhs = rhs.at[0].add(-left_bc['val'] / dx**2)
        elif left_bc_type == 'Robin':
            robin_coef = -left_bc['alpha'] / dx + left_bc['beta']
            rhs = rhs.at[0].subtract(1 / dx**2 * left_bc['val'] / robin_coef)

        if right_bc_type == 'Dirichlet':
            rhs = rhs.at[-1].add(-right_bc['val'] / dx**2)
        elif right_bc_type == 'Robin':
            robin_coef = right_bc['alpha'] / dx + right_bc['beta']
            rhs = rhs.at[-1].subtract(1 / dx**2 * right_bc['val'] / robin_coef)

        if left_bc_type == 'Dirichlet' and right_bc_type == 'Dirichlet':
            L = grid.laplacian
            phi = jnp.linalg.solve(L, rhs)
            phi = jnp.concatenate([
                jnp.array([left_bc['val']]),
                phi,
                jnp.array([right_bc['val']])
                ])

        elif left_bc_type == 'Robin' and right_bc_type == 'Dirichlet':
            L = grid.robin_dirichlet_laplacian(left_bc['alpha'], left_bc['beta'])
            phi = jnp.linalg.solve(L, rhs)
            # beta*phiL + alpha*(phi[2] - phi[1])/dx = robin_val
            #phiL = left_bc['val'] - left_bc['alpha'] * dx / (phi[1] - phi[0])
            #phi = phi.at[0]

        elif left_bc_type == 'Dirichlet' and right_bc_type == 'Robin':
            L = grid.dirichlet_robin_laplacian(right_bc['alpha'], left_bc['beta'])
            phi = jnp.linalg.solve(L, rhs)

        E = -(phi[2:] - phi[:-2]) / (2*dx)
        return E


    def poisson_solve_from_fs(self, fs, boundary_conditions):
        fe = fs['electron']
        fi = fs['ion']
        rho_c = self.plasma.Ze * self.n(fe, self.grids['electron']) + \
                self.plasma.Zi * self.n(fi, self.grids['ion'])
        E = self.poisson_solve(rho_c, boundary_conditions)
        return E


    def vlasov_fp_rhs(self, fs, boundary_conditions):
        fe = fs['electron']
        fi = fs['ion']
        E = self.poisson_solve_from_fs(fs, boundary_conditions)
        
        electron_rhs = self.vlasov_fp_single_species_rhs(fe, E, self.plasma.Ae, self.plasma.Ze, 
                                                         self.grids['electron'],
                                                         boundary_conditions['electron'])
        ion_rhs = self.vlasov_fp_single_species_rhs(fi, E, self.plasma.Ai, self.plasma.Zi, 
                                                         self.grids['ion'],
                                                         boundary_conditions['ion'])

        # TODO: implement cross-species collision term

        return dict(electron=electron_rhs, ion=ion_rhs)


    def vlasov_fp_single_species_rhs(self, f, E, A, Z, grid, bcs):
        # free streaming term
        f_bc_x = self.apply_bcs(f, bcs, 'x')

        v = jnp.expand_dims(grid.vs, axis=0)
        F = lambda left, right: jnp.where(v > 0, left * v, right * v)
        vdfdx = slope_limited_flux_divergence(f_bc_x, 'MUSCL', F, grid.dx, axis=0)

        # electrostatic acceleration term
        f_bc_v = self.apply_bcs(f, bcs, 'v')

        E = jnp.expand_dims(E, axis=1)
        fac = self.plasma.omega_c_tau * Z / A
        F = lambda left, right: jnp.where(fac * E > 0, left * fac * E, right * fac * E)
        Edfdv = slope_limited_flux_divergence(f_bc_v, 'MUSCL', F, grid.dv, axis=1)

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
