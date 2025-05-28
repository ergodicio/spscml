from functools import partial
import jax

from ..plasma import TwoSpeciesPlasma
from ..grids import Grid
from ..rk import ssprk2
from ..muscl import slope_limited_flux_divergence
from ..fulltensor_vfp.solver import Solver as KineticSolver

class Solver():
    def __init__(plasma: TwoSpeciesPlasma,
                 grid: Grid):
        self.plasma = plasma
        self.grid = grid


    @partial(jax.jit, static_argnums=(0, 3))
    def step(self, qs, dt, bcs):
        q1, E = ssprk2(qs, self.linear_density_explicit_rhs, dt, bcs)
        return self.linear_density_implicit(q1, dt)


    def solve(self, t_end, initial_conditions, bcs):
        q0 = {
            'electron': initial_conditions['electron'](self.grid.x),
            'ion': initial_conditions['ion'](self.grid.x),
        }

        t = 0.0
        dt = 0.01
        q = q0
        while t < t_end:
            dt_step = min(dt, t-t_end)
            q = self.step(q, dt)


    def linear_density_explicit_rhs(self, qs, bcs):
        Ne = q['electron']['N']
        Nue = q['electron']['Nuz']
        Ni = q['ion']['N']
        Nui = q['ion']['Nuz']

        sheathwidth = self.grid.Lx / 5
        sheathN = 100

        anode_grid = Grid(sheathN, sheathwidth)
        anode_grids = {'x': anode_grid, 'electron': anode_grid.extend_to_phase_space(30, 50),
                       'ion': anode_grid.extend_to_phase_space(30, 50)}
        anode_solver = KineticSolver(plasma, anode_grids)

        cathode_grid = Grid(sheathN, sheathwidth)
        cathode_grids = {'x': cathode_grid, 'electron': cathode_grid.extend_to_phase_space(30, 50),
                       'ion': cathode_grid.extend_to_phase_space(30, 50)}
        cathode_solver = KineticSolver(plasma, cathode_grids)

        all_dxs = jnp.concatenate(jnp.ones(100)*anode_grid.dx, 
                                  jnp.ones(self.grid.Nx, self.grid.dx),
                                  jnp.ones(100)*cathode_grid.dx)

        cathode_ion_absorbing_wall = lambda f_in: jnp.where(cathode_grids['ion'].vT > 0, 0.0, f_in)
        cathode_electron_absorbing_wall = lambda f_in: jnp.where(cathode_grids['electron'].vT > 0, 0.0, f_in)
        anode_ion_absorbing_wall = lambda f_in: jnp.where(anode_grids['ion'].vT < 0, 0.0, f_in)
        anode_electron_absorbing_wall = lambda f_in: jnp.where(anode_grids['electron'].vT < 0, 0.0, f_in)

        Te = 1.0
        Ti = 1.0
        cathode_initial_conditions = { 
            'electron': lambda x, v: Ne[0] / (jnp.sqrt(2*jnp.pi)*vte) * jnp.exp(-Ae*((v - Nue[0]/Ne[0])**2) / (2*Te)),
            'ion': lambda x, v: Ni[0] / (jnp.sqrt(2*jnp.pi)*vti) * jnp.exp(-Ai*((v - Nui[0]/Ni[0])**2) / (2*Ti))
        }
        anode_initial_conditions = { 
            'electron': lambda x, v: Ne[-1] / (jnp.sqrt(2*jnp.pi)*vte) * jnp.exp(-Ae*((v - Nue[-1]/Ne[-1])**2) / (2*Te)),
            'ion': lambda x, v: Ni[-1] / (jnp.sqrt(2*jnp.pi)*vti) * jnp.exp(-Ai*((v - Nui[-1]/Ni[-1])**2) / (2*Ti))
        }

        def step_in_unison(all_fs):
            cathode_fs, anode_fs = all_fs

            cathode_rho_c = cathode_solver.rho_c(cathode_fs)
            anode_rho_c = anode_solver.rho_c(anode_fs)
            fluid_rho_c = Ni * self.plasma.Zi + Ne * self.plasma.Ze
            all_rho_c = jnp.concatenate(cathode_rho_c, fluid_rho_c, anode_rho_c)

            rho_c_integral = jnp.cumsum(all_rho_c * all_dxs)
            rho_c_double_integral = jnp.cumsum(rho_c_integral * all_dxs)

            phiC = bcs['phi']['cathode']
            phiA = bcs['phi']['anode']

            cathode_robin_val = phiA + (-(sheathwidth + self.grid.Lx) * rho_c_integral[sheathN] \
                    + (rho_c_double_integral[-1] - rho_c_double_integral[-(sheathN+self.grid.Nx)]))

            anode_robin_val = phiC + (-(sheathwidth + self.grid.Lx) * rho_c_integral[sheathN+self.grid.Nx] \
                    + (rho_c_double_integral[sheathN+self.grid.Nx] - rho_c_double_integral[0]))

            cathode_bcs = {
                'electron': {
                    'x': {
                        'left': cathode_electron_absorbing_wall,
                        'right': lambda f_in: initial_conditions['electron'](*cathode_grids['electron'].xv)[:2, :],
                    },
                    'v': {
                        'left': jnp.zeros_like,
                        'right': jnp.zeros_like,
                    }
                },
                'ion': {
                    'x': {
                        'left': cathode_ion_absorbing_wall,
                        'right': lambda f_in: initial_conditions['ion'](*cathode_grids['ion'].xv)[:2, :],
                    },
                    'v': {
                        'left': jnp.zeros_like,
                        'right': jnp.zeros_like,
                    }
                },
                'phi': {
                    'left': {
                        'type': 'Dirichlet',
                        'val': phiC
                    },
                    'right': {
                        'type': 'Robin',
                        'alpha': (sheathwidth+self.grid.Lx),
                        'beta': 1.0,
                        'val': cathode_robin_val
                    },
                }
            }

            anode_bcs = {
                'electron': {
                    'x': {
                        'left': lambda f_in: initial_conditions['electron'](*anode_grids['electron'].xv)[:2, :],
                        'right': anode_electron_absorbing_wall,
                    },
                    'v': {
                        'left': jnp.zeros_like,
                        'right': jnp.zeros_like,
                    }
                },
                'ion': {
                    'x': {
                        'left': lambda f_in: initial_conditions['ion'](*anode_grids['ion'].xv)[:2, :],
                        'right': anode_ion_absorbing_wall,
                    },
                    'v': {
                        'left': jnp.zeros_like,
                        'right': jnp.zeros_like,
                    }
                },
                'phi': {
                    'left': {
                        'type': 'Robin',
                        'alpha': (sheathwidth+self.grid.Lx),
                        'beta': 1.0,
                        'val': anode_robin_val
                    },
                    'right': {
                        'type': 'Dirichlet',
                        'val': phiA
                    },
                }
            }

            cathode_rhs = cathode_solver.vlasov_fp_rhs(cathode_fs, cathode_bcs)
            anode_rhs = anode_solver.vlasov_fp_rhs(anode_fs, anode_bcs)
            return (cathode_fs + dt*cathode_rhs, anode_fs + dt*anode_rhs)


        cathode_f0 = {
            'electron': cathode_initial_conditions['electron'](*cathode_grids['electron'].xv),
            'ion': cathode_initial_conditions['ion'](*cathode_grids['ion'].xv),
        }
        anode_f0 = {
            'electron': anode_initial_conditions['electron'](*anode_grids['electron'].xv),
            'ion': anode_initial_conditions['ion'](*anode_grids['ion'].xv),
        }
        f0 = (cathode_f0, anode_f0)


        
    

    def linear_density_explicit_single_species_rhs(self, q, E, A, Z, flux_bcs):
        N = q['N']
        Nuz = q['Nuz']
        uz = Nuz / N


