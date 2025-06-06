import jax.numpy as jnp
import jax
import equinox as eqx

class Solver():
    def __init__(self, plasma, grids):
        self.plasma = plasma
        self.grids = grids


    def rho_c(self, fs):
        rho_e = C00 * fs['electron']['f0'] 
        rho_i = C00 * fs['ion']['f0']
        return rho_e * self.plasma.Ze + rho_i * self.plasma.Zi


    def step(self, fs, dt, bcs):
        E = poisson_solve(self.grids['x'], self.plasma, self.rho_c(fs), bcs)


    # 
    def advance_conserved_quantities_single_species(self, fs):

