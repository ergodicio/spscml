from functools import partial
import jax.numpy as jnp
import jax

from ..plasma import TwoSpeciesPlasma
from ..grids import Grid
from ..rk import ssprk2
from ..muscl import slope_limited_flux_divergence
from ..fulltensor_vfp.solver import Solver as KineticSolver
from ..utils import maxwellian_1d, zeroth_moment, first_moment
from ..poisson import poisson_solve

class Solver():
    def __init__(self, plasma: TwoSpeciesPlasma, grid):
        self.plasma = plasma
        self.grid = grid


    @partial(jax.jit, static_argnums=(0,))
    def step(self, qs, dt, bcs):
        q_rhs = self.linear_density_explicit_rhs(qs, dt, bcs)
        q1 = jax.tree.map(lambda y, dy: y + dt*dy, qs, q_rhs)
        return q1


    def solve(self, t_end, initial_conditions, bcs):
        q0 = jax.tree.map(lambda ic: ic(self.grid.xs), initial_conditions)

        t = 0.0
        dt = 0.01
        q = q0
        while t < t_end:
            dt_step = min(dt, t_end-t)
            q = self.step(q, dt_step, bcs)
            t += dt_step
            print("t = ", t)

        return q


    def linear_density_explicit_rhs(self, qs, dt, bcs):
        Ne = qs['electron']['N']
        Nue = qs['electron']['Nuz']
        Ni = qs['ion']['N']
        Nui = qs['ion']['Nuz']

        fluid_rho_c = Ni * self.plasma.Zi + Ne * self.plasma.Ze

        sheathwidth = self.grid.Lx / 5
        sheathN = 100

        anode_grid = Grid(sheathN, sheathwidth)
        anode_grids = {'x': anode_grid, 'electron': anode_grid.extend_to_phase_space(30, 50),
                       'ion': anode_grid.extend_to_phase_space(30, 50)}
        anode_solver = KineticSolver(self.plasma, anode_grids)

        cathode_grid = Grid(sheathN, sheathwidth)
        cathode_grids = {'x': cathode_grid, 'electron': cathode_grid.extend_to_phase_space(30, 50),
                       'ion': cathode_grid.extend_to_phase_space(30, 50)}
        cathode_solver = KineticSolver(self.plasma, cathode_grids)

        all_dxs = jnp.concatenate([jnp.ones(100)*anode_grid.dx, 
                                  jnp.ones(self.grid.Nx)*self.grid.dx,
                                  jnp.ones(100)*cathode_grid.dx])

        cathode_ion_absorbing_wall = lambda f_in: jnp.where(cathode_grids['ion'].vT > 0, 0.0, f_in)
        cathode_electron_absorbing_wall = lambda f_in: jnp.where(cathode_grids['electron'].vT > 0, 0.0, f_in)
        anode_ion_absorbing_wall = lambda f_in: jnp.where(anode_grids['ion'].vT < 0, 0.0, f_in)
        anode_electron_absorbing_wall = lambda f_in: jnp.where(anode_grids['electron'].vT < 0, 0.0, f_in)

        Te = 1.0
        Ti = 1.0
        cathode_initial_conditions = { 
            'electron': maxwellian_1d(self.plasma.Ae, Ne[0], Nue[0], Te),
            'ion': maxwellian_1d(self.plasma.Ai, Ni[0], Nui[0], Ti),
        }
        anode_initial_conditions = { 
            'electron': maxwellian_1d(self.plasma.Ae, Ne[-1], Nue[-1], Te),
            'ion': maxwellian_1d(self.plasma.Ai, Ni[-1], Nui[-1], Ti),
        }

        electrode_dt = dt / 1000

        def step_in_unison(all_fs, just_poisson=False):
            cathode_fs, anode_fs = all_fs

            cathode_rho_c = cathode_solver.rho_c(cathode_fs)
            anode_rho_c = anode_solver.rho_c(anode_fs)
            all_rho_c = jnp.concatenate([cathode_rho_c, fluid_rho_c, anode_rho_c])

            rho_c_integral = jnp.cumsum(all_rho_c * all_dxs)
            rho_c_double_integral = jnp.cumsum(rho_c_integral * all_dxs)

            phiC = bcs['phi']['left']
            phiA = bcs['phi']['right']

            cathode_robin_val = phiA + (-(sheathwidth + self.grid.Lx) * rho_c_integral[sheathN] \
                    + (rho_c_double_integral[-1] - rho_c_double_integral[-(sheathN+self.grid.Nx)]))

            anode_robin_val = phiC + (-(sheathwidth + self.grid.Lx) * rho_c_integral[sheathN+self.grid.Nx] \
                    + (rho_c_double_integral[sheathN+self.grid.Nx] - rho_c_double_integral[0]))

            cathode_bcs = {
                'electron': {
                    'x': {
                        'left': cathode_electron_absorbing_wall,
                        'right': lambda f_in: cathode_initial_conditions['electron'](*cathode_grids['electron'].xv)[:2, :],
                    },
                    'v': { 'left': jnp.zeros_like, 'right': jnp.zeros_like, }
                },
                'ion': {
                    'x': {
                        'left': cathode_ion_absorbing_wall,
                        'right': lambda f_in: cathode_initial_conditions['ion'](*cathode_grids['ion'].xv)[:2, :],
                    },
                    'v': { 'left': jnp.zeros_like, 'right': jnp.zeros_like, }
                },
                'phi': {
                    'left': { 'type': 'Dirichlet', 'val': phiC },
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
                        'left': lambda f_in: anode_initial_conditions['electron'](*anode_grids['electron'].xv)[:2, :],
                        'right': anode_electron_absorbing_wall,
                    },
                    'v': { 'left': jnp.zeros_like, 'right': jnp.zeros_like, }
                },
                'ion': {
                    'x': {
                        'left': lambda f_in: anode_initial_conditions['ion'](*anode_grids['ion'].xv)[:2, :],
                        'right': anode_ion_absorbing_wall,
                    },
                    'v': { 'left': jnp.zeros_like, 'right': jnp.zeros_like, }
                },
                'phi': {
                    'left': {
                        'type': 'Robin',
                        'alpha': (sheathwidth+self.grid.Lx),
                        'beta': 1.0,
                        'val': anode_robin_val
                    },
                    'right': { 'type': 'Dirichlet', 'val': phiA },
                }
            }

            if just_poisson:
                cathode_phi = cathode_solver.poisson_solve_from_fs(cathode_fs, cathode_bcs)
                anode_phi = anode_solver.poisson_solve_from_fs(anode_fs, anode_bcs)
                return (cathode_phi, anode_phi)

            print("Cathode")
            cathode_rhs = cathode_solver.vlasov_fp_rhs(cathode_fs, cathode_bcs)
            print("Anode")
            print("Anode bcs: ", anode_fs)
            anode_rhs = anode_solver.vlasov_fp_rhs(anode_fs, anode_bcs)

            return jax.tree.map(lambda y, dy: y+electrode_dt*dy,
                                (cathode_fs, anode_fs),
                                (cathode_rhs, anode_rhs))


        cathode_f0 = {
            'electron': cathode_initial_conditions['electron'](*cathode_grids['electron'].xv),
            'ion': cathode_initial_conditions['ion'](*cathode_grids['ion'].xv),
        }
        anode_f0 = {
            'electron': anode_initial_conditions['electron'](*anode_grids['electron'].xv),
            'ion': anode_initial_conditions['ion'](*anode_grids['ion'].xv),
        }
        f0 = (cathode_f0, anode_f0)
        fs = jax.lax.fori_loop(0, 1000, lambda i, fs: step_in_unison(fs), f0)
        jax.debug.print("Done with for loop")

        cathode_presheath_Ne = zeroth_moment(fs[0]['electron'], cathode_grids['electron'])[-1]
        cathode_presheath_Nue = first_moment(fs[0]['electron'], cathode_grids['electron'])[-1]
        anode_presheath_Ne = zeroth_moment(fs[1]['electron'], anode_grids['electron'])[0]
        anode_presheath_Nue = first_moment(fs[1]['electron'], anode_grids['electron'])[0]

        cathode_presheath_Ni = zeroth_moment(fs[0]['ion'], cathode_grids['ion'])[-1]
        cathode_presheath_Nui = first_moment(fs[0]['ion'], cathode_grids['ion'])[-1]
        anode_presheath_Ni = zeroth_moment(fs[1]['ion'], anode_grids['ion'])[0]
        anode_presheath_Nui = first_moment(fs[1]['ion'], anode_grids['ion'])[0]

        jax.debug.print("cathode Ne_0: {}", zeroth_moment(cathode_f0['electron'], cathode_grids['electron'])[-1])
        jax.debug.print("cathode Ne: {}", cathode_presheath_Ne)
        jax.debug.print("anode Ne: {}", anode_presheath_Ne)
        jax.debug.print("cathode Ni_0: {}", zeroth_moment(cathode_f0['ion'], cathode_grids['ion'])[-1])
        jax.debug.print("cathode Ni: {}", cathode_presheath_Ni)
        jax.debug.print("anode Ni: {}", anode_presheath_Ni)

        cathode_phi, anode_phi = step_in_unison(fs, just_poisson=True)

        phi_bcs = { 'phi': {
                'left': { 'type': 'Dirichlet', 'val': cathode_phi[-1], },
                'right': { 'type': 'Dirichlet', 'val': anode_phi[0], },
        } }
        E = poisson_solve(self.grid, self.plasma, fluid_rho_c, phi_bcs)

        electron_bcs = {
            'N': { 'left': cathode_presheath_Ne, 'right': anode_presheath_Ne },
            'Nuz': { 'left': cathode_presheath_Nue, 'right': anode_presheath_Nue },
        }
        electron_rhs = self.linear_density_explicit_single_species_rhs(
                qs['electron'], E, self.plasma.Ae, self.plasma.Ze, electron_bcs)
        ion_bcs = {
            'N': { 'left': cathode_presheath_Ni, 'right': anode_presheath_Ni },
            'Nuz': { 'left': cathode_presheath_Nui, 'right': anode_presheath_Nui },
        }
        ion_rhs = self.linear_density_explicit_single_species_rhs(
                qs['ion'], E, self.plasma.Ai, self.plasma.Zi, ion_bcs)

        return dict(electron=electron_rhs, ion=ion_rhs)
        
    

    def linear_density_explicit_single_species_rhs(self, q, E, A, Z, bcs):
        N = q['N']
        Nuz = q['Nuz']
        uz = Nuz / N

        N_bc = jnp.concatenate([jnp.array([bcs['N']['left'], bcs['N']['left']]),
                                N,
                                jnp.array([bcs['N']['right'], bcs['N']['right']])])
        Nu_bc = jnp.concatenate([jnp.array([bcs['Nuz']['left'], bcs['Nuz']['left']]),
                                 Nuz,
                                 jnp.array([bcs['Nuz']['right'], bcs['Nuz']['right']])])
        q_bc = jnp.stack([N_bc, Nu_bc], axis=1)

        def flux(left, right):
            u_left = jnp.expand_dims(left[:, 1] / left[:, 0], axis=1)
            u_right = jnp.expand_dims(right[:, 1] / right[:, 0], axis=1)
            LF_penalty = jnp.maximum(jnp.abs(u_left), jnp.abs(u_right))
            return 0.5*(left*u_left + right*u_right) - 0.5*LF_penalty*(u_right-u_left)

        rhs = slope_limited_flux_divergence(q_bc, 'MUSCL', flux, self.grid.dx, axis=0)
        print("rhs shape: ", rhs.shape)
        print("E shape: ", E.shape)
        print("N shape: ", N.shape)
        rhs = rhs.at[..., 1].add(N*Z/A*E)

        return {'N': rhs[:, 0], 'Nuz': rhs[:, 1]}
