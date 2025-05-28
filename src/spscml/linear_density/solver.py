from ..plasma import TwoSpeciesPlasma
from ..grids import Grid
from ..rk import ssprk2
from ..muscl import slope_limited_flux_divergence

class Solver():
    def __init__(plasma: TwoSpeciesPlasma,
                 grid: Grid,
                 initial_conditions):
        self.plasma = plasma
        self.grid = grid
        self.initial_conditions = initial_conditions


    def step(self, qs, dt):
        q1 = ssprk2(qs, self.linear_density_explicit_rhs, dt)
        return self.linear_density_implicit(q1, dt)


    def solve(self, t_end):
        q0 = {
            'electron': self.initial_conditions['electron'](self.grid.x),
            'ion': self.initial_conditions['ion'](self.grid.x),
        }

        t = 0.0
        dt = 0.01
        q = q0
        while t < t_end:
            dt_step = min(dt, t-t_end)
            q = self.step(q, dt)


    def linear_density_explicit_rhs(self, qs):
        
    

    def linear_density_explicit_single_species_rhs(self, q, E, A, Z, flux_bcs):
        N = q['N']
        Nuz = q['Nuz']
        uz = Nuz / N


