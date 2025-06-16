import jax.numpy as jnp
import jax
import equinox as eqx
from diffrax import diffeqsolve, Euler, Dopri5, ODETerm, PIDController, RESULTS, SaveAt

from jaxtyping import PyTree

from ..fulltensor_vfp.dougherty import lbo_operator_ij_L_diagonals
from ..plasma import TwoSpeciesPlasma
from ..rk import ssprk2
from ..muscl import slope_limited_flux_divergence
from ..poisson import poisson_solve

SPECIES = ['electron', 'ion']

class Solver(eqx.Module):
    plasma: TwoSpeciesPlasma
    r: int
    grids: PyTree
    nu_ee: float
    nu_ii: float


    def __init__(self,
                 plasma: TwoSpeciesPlasma, 
                 r: int,
                 grids,
                 nu_ee, nu_ii):
        self.plasma = plasma
        self.r = r
        self.grids = grids
        self.nu_ee = nu_ee
        self.nu_ii = nu_ii


    def solve(self, dt, Nt, initial_conditions, boundary_conditions, dtmax):
        f0 = {
            'electron': initial_conditions['electron'](*self.grids['electron'].xv),
            'ion': initial_conditions['ion'](*self.grids['ion'].xv),
        }

        solution = diffeqsolve(
            terms=ODETerm(self.step),
            solver=Stepper(),
            t0=0.0,
            t1=Nt*dt,
            max_steps=Nt + 4,
            dt0=dt,
            y0=f0,
            args={"f0": f0, "dt": dt, 'bcs': boundary_conditions},
            saveat=SaveAt(t1=True),
        )
        return jax.tree.map(lambda fs: fs[0, ...], solution.ys)


    def step(self, t, ys, args):
        ys = self.step_K(t, ys, args)
        jax.debug.print("S after K: {}", ys['electron'][1][:3, :3])
        ys = self.step_S(t, ys, args)
        jax.debug.print("S after S: {}", ys['electron'][1][:3, :3])
        ys = self.step_L(t, ys, args)
        jax.debug.print("S after L: {}", ys['electron'][1][:3, :3])
        return ys


    def step_K(self, t, ys, args):
        E = self.solve_poisson(ys, self.grids, args['bcs'])
        As = {'electron': self.plasma.Ae, 'ion': self.plasma.Ai}
        Zs = {'electron': self.plasma.Ze, 'ion': self.plasma.Zi}

        K_of = lambda X, S, V: (X.T @ S).T

        Ks = ssprk2({ sp: K_of(*ys[sp]) for sp in SPECIES },
                    lambda Ks: {
                        sp: self.K_step_single_species_nonstiff_RHS(Ks[sp], self.grids[sp],
                                                                    {**args, 'V': ys[sp][2], 'E': E, 
                                                                     'Z': Zs[sp], 'A': As[sp]})
                        for sp in SPECIES},
                    args['dt'])

        def XSV_of(K, y, grid):
            _, _, V = y
            Xt, S = jnp.linalg.qr(K.T)
            return (Xt.T / grid.dx**0.5, S * grid.dx**0.5, V)

        return { sp: XSV_of(Ks[sp], ys[sp], self.grids[sp]) for sp in SPECIES }


    def K_step_single_species_nonstiff_RHS(self, K, grid, args):
        V, E = args['V'], args['E']
        v = grid.vs
        r = self.r
        assert V.shape == (r, grid.Nv)

        v_plus_matrix = V @ jnp.diag(jnp.where(v > 0, v, 0.0)) @ V.T * grid.dv
        v_minus_matrix = V @ jnp.diag(jnp.where(v < 0, v, 0.0)) @ V.T * grid.dv

        V_left_matrix = V @ jnp.concatenate([jnp.zeros((r, 1)), jnp.diff(V, axis=1) / grid.dv], axis=1).T * grid.dv
        E_plus = jnp.atleast_2d(jnp.where(E > 0, E, 0.0))
        V_right_matrix = V @ jnp.concatenate([jnp.diff(V, axis=1) / grid.dv, jnp.zeros((r, 1))], axis=1).T * grid.dv
        E_minus = jnp.atleast_2d(jnp.where(E < 0, E, 0.0))
        #V_collisions_matrix = V @ self.apply_collisions(V, args).T * grid.dv

        K_bcs = self.apply_absorbing_wall_bcs(K, V, grid, n_ghost_cells=2)
        v_flux_func = lambda left, right: v_plus_matrix @ left + v_minus_matrix @ right
        v_flux = slope_limited_flux_divergence(K_bcs, 'minmod', v_flux_func, grid.dx, axis=1)

        fac = self.plasma.omega_c_tau * args['Z'] / args['A']
        E_flux = fac * (V_left_matrix @ (K * E_plus) + V_right_matrix @ (K * E_minus))

        # TODO: collisions
        return -v_flux - E_flux



    def step_S(self, t, ys, args):
        E = self.solve_poisson(ys, self.grids, args['bcs'])
        As = {'electron': self.plasma.Ae, 'ion': self.plasma.Ai}
        Zs = {'electron': self.plasma.Ze, 'ion': self.plasma.Zi}
        
        S_of = lambda X, S, V: S

        Ss = ssprk2({ sp: S_of(*ys[sp]) for sp in SPECIES }, 
                    lambda Ss: { 
                                sp: self.S_step_single_species_nonstiff_RHS(Ss[sp], self.grids[sp], 
                                                                            {**args, 'X': ys[sp][0], 'V': ys[sp][2], 'E': E,
                                                                             'Z': Zs[sp], 'A': As[sp]}) 
                                for sp in SPECIES}, 
                    args['dt'])

        def XSV_of(S, y):
            X, _, V = y
            return (X, S, V)
        return { sp: XSV_of(Ss[sp], ys[sp]) for sp in SPECIES }


    def S_step_single_species_nonstiff_RHS(self, S, grid, args):
        X, V, E = args['X'], args['V'], args['E']
        v = grid.vs
        r = self.r

        v_plus_matrix = V @ jnp.diag(jnp.where(v > 0, v, 0.0)) @ V.T * grid.dv
        v_minus_matrix = V @ jnp.diag(jnp.where(v < 0, v, 0.0)) @ V.T * grid.dv

        K = self.apply_absorbing_wall_bcs((X.T @ S).T, V, grid, n_ghost_cells=1)
        K_diff_left = jnp.diff(K[:, :-1], axis=1) / grid.dx
        K_left_matrix = X @ K_diff_left.T * grid.dx
        K_diff_right = jnp.diff(K[:, 1:], axis=1) / grid.dx
        K_right_matrix = X @ K_diff_right.T * grid.dx

        v_term = K_left_matrix @ v_plus_matrix.T + K_right_matrix @ v_minus_matrix.T

        V_left_matrix = V @ jnp.concatenate([jnp.zeros((r, 1)), jnp.diff(V, axis=1) / grid.dv], axis=1).T * grid.dv
        E_plus_matrix = X @ jnp.diag(jnp.where(E > 0, E, 0.0)) @ X.T * grid.dx
        V_right_matrix = V @ jnp.concatenate([jnp.diff(V, axis=1) / grid.dv, jnp.zeros((r, 1))], axis=1).T * grid.dv
        E_minus_matrix = X @ jnp.diag(jnp.where(E < 0, E, 0.0)) @ X.T * grid.dx

        fac = self.plasma.omega_c_tau * args['Z'] / args['A']
        E_term = fac * (E_plus_matrix @ S @ V_left_matrix.T + E_minus_matrix @ S @ V_right_matrix.T)

        return v_term + E_term


    def step_L(self, t, ys, args):
        E = self.solve_poisson(ys, self.grids, args['bcs'])
        As = {'electron': self.plasma.Ae, 'ion': self.plasma.Ai}
        Zs = {'electron': self.plasma.Ze, 'ion': self.plasma.Zi}

        L_of = lambda X, S, V: S @ V
        
        Ls = ssprk2({ sp: L_of(*ys[sp]) for sp in SPECIES },
                    lambda Ls: {
                        sp: self.L_step_single_species_nonstiff_RHS(Ls[sp], self.grids[sp],
                                                                    {**args, 'X': ys[sp][0], 'E': E,
                                                                     'Z': Zs[sp], 'A': As[sp]})
                        for sp in SPECIES},
                    args['dt'])

        def XSV_of(L, y, grid):
            X, _, _ = y
            (Vt, S) = jnp.linalg.qr(L.T)
            return (X, S * grid.dv**0.5, Vt.T / grid.dv**0.5)

        return { sp: XSV_of(Ls[sp], ys[sp], self.grids[sp]) for sp in SPECIES }


    def L_step_single_species_nonstiff_RHS(self, L, grid, args):
        X, E = args['X'], args['E']
        v = grid.vs
        r = self.r

        Vt, S = jnp.linalg.qr(L.T)
        S = S * grid.dv**0.5
        V = Vt.T / grid.dv**0.5

        K = self.apply_absorbing_wall_bcs((X.T @ S).T, V, grid, n_ghost_cells=1)
        K_diff_left = jnp.diff(K[:, :-1], axis=1) / grid.dx
        K_left_matrix = X @ K_diff_left.T * grid.dx
        K_diff_right = jnp.diff(K[:, 1:], axis=1) / grid.dx
        K_right_matrix = X @ K_diff_right.T * grid.dx
        
        v_plus = jnp.where(v > 0, v, 0.0)
        v_minus = jnp.where(v < 0, v, 0.0)
        v_flux = jnp.atleast_2d(v_plus) * (K_left_matrix @ V) + jnp.atleast_2d(v_minus) * (K_right_matrix @ V)
        
        fac = self.plasma.omega_c_tau * args['Z'] / args['A']
        E_plus_matrix = X @ jnp.diag(jnp.where(fac*E > 0, fac*E, 0.0)) @ X.T * grid.dx
        E_minus_matrix = X @ jnp.diag(jnp.where(fac*E < 0, fac*E, 0.0)) @ X.T * grid.dx
        L_bcs = jnp.pad(L, [(0, 0), (2, 2)], mode='empty')
        E_flux_func = lambda left, right: E_plus_matrix @ left + E_minus_matrix @ right

        E_flux = slope_limited_flux_divergence(L_bcs, 'minmod', E_flux_func, grid.dv, axis=1)

        return -v_flux - E_flux


    def solve_poisson(self, ys, grids, bcs):
        rho_c = self.rho_c_species(ys['electron'], self.plasma.Ze, grids['electron']) + \
                self.rho_c_species(ys['ion'], self.plasma.Zi, grids['ion'])
        E = poisson_solve(grids['x'], self.plasma, rho_c, bcs)
        assert E.shape == (grids['x'].Nx,)
        return E


    def rho_c_species(self, y, Z, grid):
        X, S, V = y

        V_mass_vector = V @ jnp.ones(grid.Nv) * grid.dv
        result = (X.T @ S @ V_mass_vector) * Z
        assert result.shape == (grid.Nx,)
        return result


    def apply_absorbing_wall_bcs(self, K, V, grid, n_ghost_cells):
        v = grid.vs
        V_leftgoing_matrix = V @ jnp.diag(jnp.where(v < 0, 1.0, 0.0)) @ V.T * grid.dv
        V_rightgoing_matrix = V @ jnp.diag(jnp.where(v > 0, 1.0, 0.0)) @ V.T * grid.dv

        if n_ghost_cells == 1:
            return jnp.concatenate([
                jnp.atleast_2d(V_leftgoing_matrix @ K[:, 0]).T,
                K,
                jnp.atleast_2d(V_rightgoing_matrix @ K[:, -1]).T,
            ], axis=1)
        elif n_ghost_cells == 2:
            return jnp.concatenate([
                V_leftgoing_matrix @ K[:, [1, 0]],
                K, 
                V_rightgoing_matrix @ K[:, [-1, -2]],
            ], axis=1)


    def apply_collisions(self, Vs, args):
        dl, d, du = lbo_operator_ij_L_diagonals(
                {"grid": args['grid'], "n": 1.0,"A": args['A']},
                {"T": 1.0, "u": 0.0, "lambda": 1.0},
                )
        # dl and du have zeros in the wrong place for implementing a multiplication
        dl = dl[1:]
        du = du[:-1]

        mul = lambda f: jnp.zeros(grid.Nv) \
                .at[1:].add(dl * f[:-1]) \
                .at[:].add(d*f) \
                .at[:-1].add(du * f[1:])
        return jax.vmap(mul)(Vs)


    def collision_frequency(self, args):
        L = self.grids['x'].Lx

        midpt = L/4
        # Want 10 e-foldings between the midpoint (2/3rds of the way to the sheath)
        # and the wall
        efolding_dist = (midpt/2)/20
        x = self.grids['x'].xs
        h0 = lambda x: 1 + jnp.exp((x/efolding_dist) - midpt/efolding_dist)
        h = 1 / (0.5 * (h0(x) + h0(-x)))
        return jnp.expand_dims(h, axis=1) * args['nu']


class Stepper(Euler):
    """

    :param cfg:
    """

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del solver_state, made_jump
        y1 = terms.vf(t0, y0, args | {"dt": t1 - t0})
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, RESULTS.successful
