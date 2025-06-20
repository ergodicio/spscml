import jax.numpy as jnp
import jax
import jpu
import optimistix as optx
import mlflow

from ..grids import PhaseSpaceGrid

class Solver():
    def __init__(self, R, L, C, Lp, V0, Lz, N, mlflow_run_id):
        '''
        params:
            R: The circuit resistance [ohms]
            L: The circuit inductance [henrys]
            C: The capacitance [farads]
            Lp: The plasma inductance [henrys]
            V0: The initial capacitor voltage [volts]
            Lz: The inter-electrode length [meters]
            N: The plasma linear density [meters^-1]
        '''
        self.R = R
        self.L = L
        self.C = C
        self.Lp = Lp
        self.V0 = V0
        self.Lz = Lz
        self.N = N
        self.ureg = jpu.UnitRegistry()
        self.rootfinder = optx.Newton(rtol=1e-8, atol=1e-8)
        self.mlflow_run_id = mlflow_run_id


    def solve(self, dt, Nt, ics, sheath_solve):
        '''
        Solve the whole-device model ODE with the given dt and number of timesteps Nt.

        params:
            dt: The timestep to use [seconds]
            Nt: The number of timesteps
            ics: The ODE initial condition, with elements
                [Q0, Ip0, T0, n0], where
                - Q0 is the initial capacitor charge [coulombs]
                - Ip0 is the initial plasma current [amperes]
                - T0 is the initial temperature [eV]
                - n0 is the initial volumetric density [meters^-3]
            sheath_solve: a Callable that accepts (V, N, T), where
                - V is the plasma gap voltage [volts]
                - T is the plasma temperature [eV]
                and returns the plasma current in amperes
        '''

        @jax.jit
        def scanner(carry, ys):
            y, Vp, t = carry
            jax.debug.print("t = {}", t)
            jax.debug.print("y: {}", y)
            assert len(y) == 4
            jax.debug.callback(self.log_progress, t, y, Vp)
            Q, I, T, n = y
            Q_I = jnp.array([Q, I])
            Q_I_new, Vnew = self.implicit_euler_step(Q_I, Vp, T, n, dt, sheath_solve)
            I_new = Q_I_new[1]
            T_prime = self.step_heating_and_cooling(I_new, T, n, dt)

            T_new = (I_new / I)**2 * T
            n_new = (T_new / T_prime)**(3/2) * n

            ynew = jnp.append(Q_I_new, jnp.array([T_new, n_new]))

            return (ynew, Vnew, t+dt), jnp.append(ynew, jnp.array([Vnew, t+dt]))

        return jax.lax.scan(scanner, (ics, 300.0, 0), jnp.arange(Nt))


    def estimate_max_dt(self):
        """
        Estimates the maximum dt we should take by computing the eigenvalues
        of the linear approximation to the problem which assumes no plasma resistance.
        """
        def forward_euler_rhs(y):
            Qn, Qdotn = y
            return jnp.array([Qdotn,
                              (1 / (self.L - self.Lp) * (-Qn/self.C - self.R*Qdotn))])

        J = jax.jacobian(forward_euler_rhs)(jnp.array([1.0, 0.0]))
        lambda_max = jnp.max(jnp.abs(jnp.linalg.eigvals(J)))
        return 0.2 / lambda_max


    def implicit_euler_step(self, y, Vp, T, n, dt, sheath_solve):
        Qn, Qdotn = y

        def residual_helper(y, Ip):
            Qnext, Vpnext = y
            factor = self.Lp / (self.L - self.Lp)
            V_Rp = (1 - factor) * (Vpnext - factor * (-Qnext / self.C - self.R * Ip))
            r = jnp.array([
                Qnext - Qn - dt * Ip,
                -Ip + Qdotn + dt/(self.L - self.Lp) * (-Qnext/self.C - self.R*Ip + V_Rp)
                ])
            return r

        def residual(y, args):
            Qnext, Vpnext = y
            Ip = sheath_solve(Vpnext, T, n)
            return residual_helper(y, Ip)


        guess = jnp.array([Qn, Vp])

        Q, V = optx.root_find(residual, self.rootfinder, guess, max_steps=5, throw=False).value
        Ip = sheath_solve(V, T, n)
        return jnp.array([Q, Ip]), V


    def estimate_plasma_current(self, Vp):
        return {"Ip": -jnp.tanh(Vp / 2.5e4) * 1e6}


    def log_progress(self, t, y, Vp):
        step_ns = int(t * 1e9)
        Q, I, T, n = y
        mlflow.log_metric("Time - seconds", t, step=step_ns, run_id=self.mlflow_run_id)
        mlflow.log_metric("Capacitor charge - coulombs", Q, step=step_ns, run_id=self.mlflow_run_id)
        mlflow.log_metric("Current - amperes", I, step=step_ns, run_id=self.mlflow_run_id)
        mlflow.log_metric("Temperature - eV", T, step=step_ns, run_id=self.mlflow_run_id)
        mlflow.log_metric("Density - per cubic meter", n, step=step_ns, run_id=self.mlflow_run_id)
        mlflow.log_metric("Voltage - volts", Vp, step=step_ns, run_id=self.mlflow_run_id)


    def step_heating_and_cooling(self, I, T, n, dt):
        ureg = self.ureg
        Lz = self.Lz * ureg.m
        N = self.N * ureg.m**-1
        n = n * ureg.m**-3
        T = T * ureg.eV
        I = I * ureg.A

        eta = 1 / 1.96 * jnp.sqrt(2) * ureg.m_e**0.5 * ureg.e**2 * 10 \
                / (12 * jnp.pi**1.5 * ureg.epsilon_0**2 * T**1.5)
        a = ((N / jnp.pi / n)**0.5).to(ureg.m)
        j = I / (jnp.pi * a**2)

        dT_eta = (eta * j**2 / n).to(ureg.eV / ureg.s)

        P_br = (1.06e-19 * n.magnitude**2 * T.magnitude**0.5) * (ureg.eV / ureg.s / ureg.m**3)
        dT_rad = (P_br / n).to(ureg.eV / ureg.s)

        T_prime = (T + dt * ureg.s * (dT_eta - dT_rad)).to(ureg.eV).magnitude
        return T_prime
