import jax.numpy as jnp
import jax
import equinox as eqx
from diffrax import diffeqsolve, Euler, Dopri5, ODETerm, PIDController, RESULTS, SaveAt

from jaxtyping import PyTree

from ..plasma import TwoSpeciesPlasma
from ..rk import rk1, ssprk2
from ..muscl import slope_limited_flux, slope_limited_flux_divergence
from ..poisson import poisson_solve
from ..collisions import collision_frequency_shape_func

SPECIES = ['electron', 'ion']

SCHEME = 'upwind'

class Solver(eqx.Module):
    plasma: TwoSpeciesPlasma
    r: int
    grids: PyTree
    nu_ee: float
    nu_ii: float
    boundary_type: str
    As: dict
    Zs: dict
    nus: dict


    def __init__(self,
                 plasma: TwoSpeciesPlasma, 
                 r: int,
                 grids,
                 nu_ee, nu_ii,
                 boundary_type='AbsorbingWall'):
        self.plasma = plasma
        self.r = r
        self.grids = grids
        self.nu_ee = nu_ee
        self.nu_ii = nu_ii
        self.boundary_type = boundary_type
        self.As = {'electron': self.plasma.Ae, 'ion': self.plasma.Ai}
        self.Zs = {'electron': self.plasma.Ze, 'ion': self.plasma.Zi}
        self.nus = {'electron': self.nu_ee, 'ion': self.nu_ii}


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
            saveat=SaveAt(ts=jnp.linspace(0.0, Nt*dt, 100)),
        )
        return jax.tree.map(lambda fs: fs[-1, ...], solution.ys)


    def step(self, t, ys, args):
        dt = args['dt']
        half_dt_args = {**args, 'dt': dt/2}
        ys = self.step_K(t, ys, args)
        ys = self.step_S(t, ys, args)
        ys = self.step_L(t, ys, args)
        return ys


    def step_K(self, t, ys, args):
        K_of = lambda X, S, V: (X.T @ S).T

        flux_out = self.ion_flux_out(ys)
        args_of = lambda sp: {**args, 'V': ys[sp][2], 'Z': self.Zs[sp], 'A': self.As[sp], 'nu': self.nus[sp],
                              'flux_out': flux_out}

        def step_Ks_with_E_RHS(Ks):
            E = self.solve_poisson_KV(Ks, ys, self.grids, args['bcs'])
            return { sp: self.K_step_single_species_RHS(Ks[sp], self.grids[sp], 
                                                                 {**args_of(sp), 'E': E})
                    for sp in SPECIES }


        Ks = rk1({ sp: K_of(*ys[sp]) for sp in SPECIES }, step_Ks_with_E_RHS, args['dt'])

        def XSV_of(K, y, grid):
            _, _, V = y
            Xt, S = jnp.linalg.qr(K.T)
            return (Xt.T / grid.dx**0.5, S * grid.dx**0.5, V)

        return { sp: XSV_of(Ks[sp], ys[sp], self.grids[sp]) for sp in SPECIES }


    def K_step_single_species_RHS(self, K, grid, args):
        V, E = args['V'], args['E']
        v = grid.vs
        r = self.r
        assert V.shape == (r, grid.Nv)

        v_plus_matrix = V @ jnp.diag(jnp.where(v > 0, v, 0.0)) @ V.T * grid.dv
        v_minus_matrix = V @ jnp.diag(jnp.where(v < 0, v, 0.0)) @ V.T * grid.dv

        V_left_matrix = V @ jnp.concatenate([jnp.zeros((r, 1)), jnp.diff(V, axis=1) / grid.dv], axis=1).T * grid.dv
        V_right_matrix = V @ jnp.concatenate([jnp.diff(V, axis=1) / grid.dv, jnp.zeros((r, 1))], axis=1).T * grid.dv

        K_bcs = self.apply_K_bcs(K, V, grid, n_ghost_cells=2)
        v_flux_func = lambda left, right: v_plus_matrix @ left + v_minus_matrix @ right
        v_flux = slope_limited_flux(K_bcs, SCHEME, v_flux_func, grid.dx, axis=1)

        v_flux_diff = jnp.diff(v_flux, axis=1) / grid.dx

        fac = self.plasma.omega_c_tau * args['Z'] / args['A']
        E_plus = jnp.atleast_2d(jnp.where(fac*E > 0, fac*E, 0.0))
        E_minus = jnp.atleast_2d(jnp.where(fac*E < 0, fac*E, 0.0))
        E_flux = (V_left_matrix @ (K * E_plus) + V_right_matrix @ (K * E_minus))

        gamma = args['flux_out'] * self.flux_source_shape_func()

        n = (K.T @ (V @ jnp.ones(grid.Nv)) * grid.dv).T
        nu = args['nu'] * collision_frequency_shape_func(self.grids)
        M = self.maxwellian(grid, args)
        VM = V @ M * grid.dv

        collision_term = (n*nu+gamma)[None, :] * VM[:, None] - K * nu[None, :]

        return -v_flux_diff - E_flux + collision_term


    def step_S(self, t, ys, args):
        flux_out = self.ion_flux_out(ys)
        S_of = lambda X, S, V: S
        args_of = lambda sp: {**args, 'X': ys[sp][0], 'V': ys[sp][2],
                              'Z': self.Zs[sp], 'A': self.As[sp], 'nu': self.nus[sp], 
                              'flux_out': flux_out}

        def step_Ss_with_E_RHS(Ss):
            E = self.solve_poisson_XSV(Ss, ys, self.grids, args['bcs'])
            return { sp: self.S_step_single_species_RHS(Ss[sp], self.grids[sp], 
                                                                 {**args_of(sp), 'E': E})
                    for sp in SPECIES }


        Ss = rk1({ sp: S_of(*ys[sp]) for sp in SPECIES }, step_Ss_with_E_RHS, args['dt'])

        def XSV_of(S, y):
            X, _, V = y
            return (X, S, V)
        return { sp: XSV_of(Ss[sp], ys[sp]) for sp in SPECIES }


    def S_step_single_species_RHS(self, S, grid, args):
        X, V, E = args['X'], args['V'], args['E']
        v = grid.vs
        r = self.r

        v_plus_matrix = V @ jnp.diag(jnp.where(v > 0, v, 0.0)) @ V.T * grid.dv
        v_minus_matrix = V @ jnp.diag(jnp.where(v < 0, v, 0.0)) @ V.T * grid.dv

        K = self.apply_K_bcs((X.T @ S).T, V, grid, n_ghost_cells=1)
        K_diff_left = jnp.diff(K[:, :-1], axis=1) / grid.dx
        K_left_matrix = X @ K_diff_left.T * grid.dx
        K_diff_right = jnp.diff(K[:, 1:], axis=1) / grid.dx
        K_right_matrix = X @ K_diff_right.T * grid.dx

        v_term = K_left_matrix @ v_plus_matrix.T + K_right_matrix @ v_minus_matrix.T

        V_left_matrix = V @ jnp.concatenate([jnp.zeros((r, 1)), jnp.diff(V, axis=1) / grid.dv], axis=1).T * grid.dv
        V_right_matrix = V @ jnp.concatenate([jnp.diff(V, axis=1) / grid.dv, jnp.zeros((r, 1))], axis=1).T * grid.dv

        fac = self.plasma.omega_c_tau * args['Z'] / args['A']
        E_plus_matrix = X @ jnp.diag(jnp.where(fac*E > 0, fac*E, 0.0)) @ X.T * grid.dx
        E_minus_matrix = X @ jnp.diag(jnp.where(fac*E < 0, fac*E, 0.0)) @ X.T * grid.dx
        E_term = (E_plus_matrix @ S @ V_left_matrix.T + E_minus_matrix @ S @ V_right_matrix.T)

        gamma = args['flux_out'] * self.flux_source_shape_func()

        n = (X.T @ S @ (V @ jnp.ones(grid.Nv)) * grid.dv).T
        nu = args['nu'] * collision_frequency_shape_func(self.grids)
        X_nu_gamma_vec = X @ (n*nu + gamma) * grid.dx
        X_nu_matrix = X @ jnp.diag(nu) @ X.T * grid.dx
        M = self.maxwellian(grid, args)
        VM = V @ M * grid.dv
        collision_term = X_nu_gamma_vec[:, None] * VM[None, :] - X_nu_matrix @ S

        return v_term + E_term - collision_term


    def step_L(self, t, ys, args):
        L_of = lambda X, S, V: S @ V
        flux_out = self.ion_flux_out(ys)
        args_of = lambda sp: {**args, 'X': ys[sp][0], 'Z': self.Zs[sp], 'A': self.As[sp], 
                              'nu': self.nus[sp], 'flux_out': flux_out}

        def step_Ls_with_E_RHS(Ls):
            E = self.solve_poisson_XL(Ls, ys, self.grids, args['bcs'])
            return { sp: self.L_step_single_species_RHS(Ls[sp], self.grids[sp], 
                                                                 {**args_of(sp), 'E': E})
                    for sp in SPECIES }
        
        Ls = rk1({ sp: L_of(*ys[sp]) for sp in SPECIES }, step_Ls_with_E_RHS, args['dt'])

        def XSV_of(L, y, grid):
            X, _, _ = y
            (Vt, St) = jnp.linalg.qr(L.T)
            return (X, St.T * grid.dv**0.5, Vt.T / grid.dv**0.5)

        return { sp: XSV_of(Ls[sp], ys[sp], self.grids[sp]) for sp in SPECIES }


    def L_step_single_species_RHS(self, L, grid, args):
        X, E = args['X'], args['E']
        v = grid.vs
        r = self.r

        Vt, St = jnp.linalg.qr(L.T)
        S = St.T * grid.dv**0.5
        V = Vt.T / grid.dv**0.5

        K = self.apply_K_bcs((X.T @ S).T, V, grid, n_ghost_cells=1)
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
        E_flux = slope_limited_flux_divergence(L_bcs, SCHEME, E_flux_func, grid.dv, axis=1)

        gamma = args['flux_out'] * self.flux_source_shape_func()

        n = (X.T @ (L @ jnp.ones(grid.Nv)) * grid.dv).T
        nu = args['nu'] * collision_frequency_shape_func(self.grids)
        X_nu_gamma_vec = X @ (gamma + n*nu) * grid.dx
        X_nu_matrix = X @ jnp.diag(nu) @ X.T * grid.dx
        M = self.maxwellian(grid, args)

        collision_term = X_nu_gamma_vec[:, None] * M[None, :] - X_nu_matrix @ L

        return -v_flux - E_flux + collision_term


    def solve_poisson_ys(self, ys, grids, bcs):
        rho_c = self.rho_c_species_XSV(*ys['electron'], 
                                      self.plasma.Ze, grids['electron']) + \
                self.rho_c_species_XSV(*ys['ion'],
                                      self.plasma.Zi, grids['ion'])
        return poisson_solve(grids['x'], self.plasma, rho_c, bcs)


    def solve_poisson_KV(self, Ks, ys, grids, bcs):
        rho_c = self.rho_c_species_KV(Ks['electron'], ys['electron'][2], 
                                      self.plasma.Ze, grids['electron']) + \
                self.rho_c_species_KV(Ks['ion'], ys['ion'][2], self.plasma.Zi, grids['ion'])
        return poisson_solve(grids['x'], self.plasma, rho_c, bcs)


    def rho_c_species_KV(self, K, V, Z, grid):
        V_mass_vector = V @ jnp.ones(grid.Nv) * grid.dv
        return (K.T @ V_mass_vector) * Z


    def solve_poisson_XSV(self, Ss, ys, grids, bcs):
        rho_c = self.rho_c_species_XSV(ys['electron'][0], Ss['electron'], ys['electron'][2],
                                      self.plasma.Ze, grids['electron']) + \
                self.rho_c_species_XSV(ys['ion'][0], Ss['ion'], ys['ion'][2], 
                                      self.plasma.Zi, grids['ion'])
        return poisson_solve(grids['x'], self.plasma, rho_c, bcs)


    def rho_c_species_XSV(self, X, S, V, Z, grid):
        V_mass_vector = V @ jnp.ones(grid.Nv) * grid.dv
        return (X.T @ S @ V_mass_vector) * Z


    def solve_poisson_XL(self, Ls, ys, grids, bcs):
        rho_c = self.rho_c_species_XL(ys['electron'][0], Ls['electron'], 
                                      self.plasma.Ze, grids['electron']) + \
                self.rho_c_species_XL(ys['ion'][0], Ls['ion'], self.plasma.Zi, grids['ion'])
        return poisson_solve(grids['x'], self.plasma, rho_c, bcs)


    def rho_c_species_XL(self, X, L, Z, grid):
        L_mass_vector = L @ jnp.ones(grid.Nv) * grid.dv
        return X.T @ L_mass_vector * Z

    


    def flux_source_shape_func(self):
        Ls = self.grids['x'].Lx / 4
        return (1 / Ls - jnp.abs(self.grids['x'].xs) / Ls**2)


    def ion_flux_out(self, ys):
        X, S, V = ys['ion']
        K = (X.T @ S).T
        v = self.grids['ion'].vs
        v_vec = V @ self.grids['ion'].vs * self.grids['ion'].dv
        flux_out = -(K[:, 0]).T @ v_vec + (K[:, -1]).T @ v_vec
        return flux_out


    def apply_K_bcs(self, K, V, grid, n_ghost_cells):
        v = grid.vs
        V_leftgoing_matrix = V @ jnp.diag(jnp.where(v < 0, 1.0, 0.0)) @ V.T * grid.dv
        V_rightgoing_matrix = V @ jnp.diag(jnp.where(v > 0, 1.0, 0.0)) @ V.T * grid.dv

        if self.boundary_type == 'AbsorbingWall':
            if n_ghost_cells == 1:
                return jnp.concatenate([
                    jnp.atleast_2d(V_leftgoing_matrix @ K[:, 0]).T,
                    K,
                    jnp.atleast_2d(V_rightgoing_matrix @ K[:, -1]).T,
                ], axis=1)
            elif n_ghost_cells == 2:
                K_out_left = K[:, [0, 1]] - 2*(K[:, [1]] - K[:, [0]])
                return jnp.concatenate([
                    V_leftgoing_matrix @ K_out_left,
                    K, 
                    V_rightgoing_matrix @ K[:, [-1, -1]],
                ], axis=1)
        elif self.boundary_type == 'Periodic':
            if n_ghost_cells == 1:
                return jnp.concatenate([
                    jnp.atleast_2d(K[:, -1]).T,
                    K,
                    jnp.atleast_2d(K[:, 0]).T,
                ], axis=1)
            elif n_ghost_cells == 2:
                return jnp.concatenate([
                    K[:, [-2, -1]],
                    K, 
                    K[:, [0, 1]],
                ], axis=1)


    def maxwellian(self, grid, args):
        v = grid.vs
        T = 1.0
        n = 1.0
        theta = T / args['A']
        M = n / (jnp.sqrt(2*jnp.pi*theta)) * jnp.exp(-v**2 / (2*theta))
        return M


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
