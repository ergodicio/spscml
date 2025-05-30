import jax.numpy as jnp
import jax

from ..plasma import TwoSpeciesPlasma
from ..grids import PhaseSpaceGrid
from fulltensor_fvp.sheath_model import calculate_plasma_current

class Solver():
    
    def __init__(self, plasma, R, L, C, Lp):
        self.plasma = plasma
        self.R = R
        self.L = L
        self.C = C
        self.Lp = Lp


    def solve(self, dt, Nt, ics):
        def dydt(y):
            Q = y[0]
            Qdot = y[1]
            V_Rp = self.estimate_resistive_plasma_voltage(Qdot)

        t = 0.0


    def estimate_resistive_plasma_voltage(self, Qdot):
        V_Rp_guess = 0.0

        def residual(V_Rp):
            V_total = V_Rp + self.Lp * Qdot
            calculate_plasma_current(V_total) - Qdot

        vg = jax.value_and_grad(residual)
        
        for i in range(5):
            r_actual, r_grad = vg(V_Rp_guess)
            V_Rp_guess = V_Rp_guess - r_actual / r_grad


