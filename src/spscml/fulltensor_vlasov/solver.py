import jax.numpy as jnp
import jax
import equinox as eqx

from jaxtyping import PyTree
from typing import Callable
from functools import partial
import diffrax
from diffrax import diffeqsolve, Euler, Dopri5, ODETerm, PIDController, RESULTS, SaveAt

from ..plasma import TwoSpeciesPlasma
from ..grids import PhaseSpaceGrid
from ..rk import rk1, ssprk2, imex_ssp2, imex_euler
from ..muscl import slope_limited_flux_divergence
from ..poisson import poisson_solve
from ..utils import zeroth_moment, first_moment, second_moment, maxwellian_1d
from ..collisions_and_sources import flux_source_shape_func

class Solver(eqx.Module):
    plasma: TwoSpeciesPlasma
    grids: PyTree
    flux_source_enabled: bool
    nu_ee: float
    nu_ii: float
    adjoint_method: str

    """
    Solves the Vlasov-BGK equation
    """
    def __init__(self,
                 plasma: TwoSpeciesPlasma, 
                 grids,
                 flux_source_enabled,
                 nu_ee, 
                 nu_ii, 
                 adjoint_method=None):
        self.plasma = plasma
        self.grids = grids
        self.flux_source_enabled = flux_source_enabled
        self.nu_ee = nu_ee
        self.nu_ii = nu_ii
        self.adjoint_method = 'vjp' if adjoint_method is None else adjoint_method


    def step(self, t, fs, args):
        rhs = lambda f: self.vlasov_rhs(f, args["bcs"], args["f0"])
        return ssprk2(fs, rhs, args["dt"])


    def solve(self, dt, Nt, initial_conditions, boundary_conditions, dtmax):
        f0 = {
            'electron': initial_conditions['electron'](*self.grids['electron'].xv),
            'ion': initial_conditions['ion'](*self.grids['ion'].xv),
        }

        # Select the automatic differentiation method used by diffrax.
        # This is necessary because a forward-mode Jacobian-Vector-Product ('jvp') is incompatible
        # with the RecursiveCheckpointAdjoint method we use for reverse-mode differentiation.
        if self.adjoint_method == 'jvp':
            adjoint = diffrax.ForwardMode()
        else:
            adjoint = diffrax.RecursiveCheckpointAdjoint()

        solution = diffeqsolve(
            terms=ODETerm(self.step),
            solver=Stepper(),
            t0= 0.0,
            t1=Nt * dt,
            max_steps= Nt + 4,
            dt0=dt,
            y0=f0,
            args={"bcs": boundary_conditions, "f0": f0, "dt": dt},
            saveat=SaveAt(t1=True),
            adjoint=adjoint,
        )
        return jax.tree.map(lambda fs: fs[0, ...], solution.ys)


    def vlasov_rhs(self, fs, boundary_conditions, f0):
        fe = fs['electron']
        fi = fs['ion']
        # HACKATHON: Solve poisson equation for E
        # See poisson.py -- poisson_solve()
    
        #E = jnp.zeros(self.grids['x'].Nx)
        rho_c = self.plasma.Ze * zeroth_moment(fe,self.grids['electron']) + \
                self.plasma.Zi * zeroth_moment(fi,self.grids['ion'])
        E = poisson_solve(self.grids['x'],self.plasma,rho_c, boundary_conditions)
        electron_rhs = self.vlasov_fp_single_species_rhs(fe, E, self.plasma.Ae, self.plasma.Ze, 
                                                         self.grids['electron'],
                                                         boundary_conditions['electron'], self.nu_ee)
        ion_rhs = self.vlasov_fp_single_species_rhs(fi, E, self.plasma.Ai, self.plasma.Zi, 
                                                         self.grids['ion'],
                                                         boundary_conditions['ion'], self.nu_ii)
    
        if self.flux_source_enabled:
            ion_particle_flux = first_moment(fi, self.grids['ion'])
            total_ion_wall_flux = -ion_particle_flux[0] + ion_particle_flux[-1]

            flux_source_weight = jnp.maximum(flux_source_shape_func(self.grids['x']), 0.0)[:, None]
            fe0 = jnp.expand_dims(f0['electron'][self.grids['x'].Nx // 2, :], axis=0)
            fi0 = jnp.expand_dims(f0['ion'][self.grids['x'].Nx // 2, :], axis=0)

            electron_rhs = electron_rhs + total_ion_wall_flux * flux_source_weight * fe0
            ion_source = total_ion_wall_flux * flux_source_weight * fi0
            ion_rhs = ion_rhs + total_ion_wall_flux * flux_source_weight * fi0

        assert electron_rhs.shape == fe.shape

        return dict(electron=electron_rhs, ion=ion_rhs)




    def vlasov_fp_single_species_rhs(self, f, E, A, Z, grid, bcs, nu):
        # free streaming term
        f_bc_x = self.apply_bcs(f, bcs, 'x')

        v = jnp.expand_dims(grid.vs, axis=0)
        F = lambda left, right: jnp.where(v > 0, left * v, right * v)
        vdfdx = slope_limited_flux_divergence(f_bc_x, 'minmod', F, 
                                              grid.dx,
                                              axis=0)

        # HACKATHON: implement E*df/dv term
        f_bc_v = self.apply_bcs(f,bcs,'v')
        E = jnp.expand_dims(E,axis=1)
        fac=self.plasma.omega_c_tau * Z/A
        F = lambda left, right: jnp.where(fac * E > 0, left * fac * E, right * fac * E)
        Edfdv = slope_limited_flux_divergence(f_bc_v, 'minmod', F, 
                                              grid.dv,
                                              axis=1)
        n = self.plasma.Ze * zeroth_moment(f,grid) 
        nu = nu * jmp.ones(n.shape)
    
       # HACKATHON: implement BGK collision term
        T = 1
        M = maxwellian_1d(A,n,nu,T)
        
     
        col = (n[:,None]*M[None,:]-f)
        return -vdfdx - Edfdv + col


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


class Stepper(Euler):
    """

    :param cfg:
    """

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del solver_state, made_jump
        y1 = terms.vf(t0, y0, args | {"dt": t1 - t0})
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, RESULTS.successful
