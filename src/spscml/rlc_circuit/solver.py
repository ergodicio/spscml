import jax.numpy as jnp
import jax

from ..plasma import TwoSpeciesPlasma
from ..grids import PhaseSpaceGrid
from ..fulltensor_vfp.sheath_model import calculate_plasma_current

class Solver():
    
    def __init__(self, plasma, R, L, C, Lp, V0):
        self.plasma = plasma
        self.R = R
        self.L = L
        self.C = C
        self.Lp = Lp
        self.V0 = V0


    def solve(self, dt, Nt, ics):
        @jax.jit
        def dydt(y):
            Q = y[0]
            Qdot = y[1]
            Qdotdot = self.estimate_Qdotdot(Q, Qdot)
            return jnp.array([Qdot, Qdotdot])

        def scanner(carry, ys):
            y, Vp = carry
            ynew, Vnew = self.implicit_euler_step(y, Vp, dt)
            jax.debug.print("y: {}", ynew)
            return (ynew, Vnew), ynew

        return jax.lax.scan(scanner, (ics, 0.5), jnp.arange(Nt))


    def implicit_euler_step(self, y, Vp, dt):
        Qn, Qdotn = y

        def residual(y):
            Qnext, Vpnext = y
            Ip = self.estimate_plasma_current(Vpnext)
            factor = self.Lp / (self.L - self.Lp)
            V_Rp = (1 - factor) * (Vpnext - factor * (-Qnext / self.C - self.R * Ip))
            r = jnp.array([
                Qnext - Qn - dt * Ip,
                -Ip + Qdotn + dt/(self.L - self.Lp) * (-Qnext/self.C - self.R*Ip + V_Rp)
                ])
            return r

        jac = jax.jacobian(residual)

        guess = jnp.array([Qn, Vp])

        for i in range(5):
            r_val = residual(guess)
            J = jac(guess)
            step = -jnp.linalg.solve(J, r_val)
            step = step.at[1].set(jnp.sign(step[1]) * jnp.minimum(jnp.abs(step[1]), 0.1))
            guess = guess - jnp.linalg.solve(J, r_val)
    

        Q, V = guess
        Ip = self.estimate_plasma_current(V)
        jax.debug.print("Ip: {}, Vp: {}", Ip, V)
        return jnp.array([Q, Ip]), V


    def estimate_plasma_current(self, Vp):
        return -jnp.tanh(Vp) * .5
        #return calculate_plasma_current(Vp)
