from ..plasma import TwoSpeciesPlasma
from ..grids import PhaseSpaceGrid
from ..rk import ssprk2
from ..muscl import slope_limited_flux_divergence

class Solver():
    """
    Solves the Vlasov-Fokker-Planck equation
    """
    def __init__(plasma: TwoSpeciesPlasma, 
                 grid: PhaseSpaceGrid,
                 initial_conditions,
                 boundary_conditions):
        self.plasma = plasma
        self.grid = grid
        self.initial_conditions = initial_conditions
        self.boundary_conditions = boundary_conditions


    def max_dt(self):
        CFL = 0.8
        free_streaming_limit = CFL * self.grid.dx / self.grid.vmax
        omega_pe = self.plasma.omega_p_tau / sqrt(self.plasma.Ae)
        omega_pe_limit = 0.2 / omega_pe
        return min(omega_pe_limit, free_streaming_limit)


    def step(self, fs, dt):
        return ssprk2(fs, self.vlasov_fp_rhs, dt)


    def solve(self, t_end):
        f0 = {
            'electron': self.initial_conditions['electron'](self.grid.xv),
            'ion': self.initial_conditions['ion'](self.grid.xv),
        }

        t = 0.0
        dt = self.max_dt()
        f = f0
        while t < t_end:
            dt_step = min(dt, t - t_end)
            self.step()


    def n(self, f):
        return jnp.sum(f, axis=1) * self.grid.dv


    def poisson_solve(self, rho_c):
        """
        Solves the Poisson equation nabla^2 phi = -omega_c_tau / omega_p_tau^2 * rho_c.

        Returns E = -nabla phi
        """
        oct = self.plasma.omega_c_tau
        opt = self.plasma.omega_p_tau
        rhs = -oct / opt**2 * rho_c
        
        phi_L = self.boundary_conditions['phi']['left']
        phi_R = self.boundary_conditions['phi']['right']
        dx = self.grid.dx
        # Apply phi boundary conditions
        rhs = rhs.at[0].add(-phi_L / dx**2) \
                .at[-1].add(-phi_R / dx**2)

        L = self.grid.laplacian
        phi = jnp.linalg.solve(L, rhs).at[0].set(phi_L) \
                .at[-1].set(phi_R)
        E = -(phi[2:] - phi[:-2]) / (2*dx)
        return E


    def vlasov_fp_rhs(fs):
        fe = fs['electron']
        fi = fs['ion']
        rho_c = self.plasma.Ze * self.n(fe) + self.plasma.Zi * self.n(fi)
        E = self.poisson_solve(rho_c)
        
        electron_rhs = self.vlasov_fp_single_species_rhs(fe, E, self.plasma.Ae, self.plasma.Ze, 
                                                         self.boundary_conditions['electron'])
        ion_rhs = self.vlasov_fp_single_species_rhs(fi, E, self.plasma.Ai, self.plasma.Zi, 
                                                         self.boundary_conditions['ion'])

        # TODO: implement cross-species collision term

        return dict(electron=electron_rhs, ion=ion_rhs)


    def vlasov_fp_single_species_rhs(f, E, A, Z, bcs):
        # free streaming term
        f_bc_x = self.apply_bcs(f, bcs, 'x')

        v = jnp.expand_dims(self.grid.vs, axis=0)
        F = lambda left, right: jnp.where(v > 0, left * v, right * v)
        vdfdx = slope_limited_flux_divergence(f_bc_x, 'muscl', F, self.grid.dx, axis=0)

        # electrostatic acceleration term
        f_bc_v = self.apply_bcs(f, bcs, 'v')

        fac = self.plasma.omega_c_tau * Z / A
        F = lambda left, right: jnp.where(fac * E > 0, left * fac * E, right * fac * E, axis=1)
        Edfdv = slope_limited_flux_divergence(f_bc_v, 'muscl', F, self.grid.dv, axis=1)

        # TODO: implement Fokker-Planck operator

        return -vdfdx - Edfdv


    def apply_bcs(f, bcs, dim):
        bc = bcs[dim]
        if dim == 'x':
            axis = 0
        elif dim == 'v':
            axis = 1

        if axis == 0:
            left = bc['left'](f[0:2, :])
            right = bc['right'](f[-2:, :])
        elif axis == 1:
            left = bc['left'](f[:, 0:2])
            right = bc['right'](f[:, -2:])

        return jnp.concatenate([left, f, right], axis=axis)
