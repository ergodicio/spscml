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
            sheath_solve: a Callable that accepts (V, T, n), where
                - V is the plasma gap voltage [volts]
                - T is the plasma temperature [eV]
                - n is the plasma density [m^-3]
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

            # Adiabatic compression or expansion: T_new is determined only from the change in
            # current, per the Bennett relation describing thermal equilibrium:
            #
            #               (1+Z)*N*T = mu_0 * I_p^2 / (8pi)
            #
            T_new = (I_new / I)**2 * T
            # During the compression or expansion to thermal equilibrium, n changes adiabatically.
            # The adiabat it follows is now based at T_prime, not T^n.
            n_new = (T_new / T_prime)**(3/2) * n

            ynew = jnp.append(Q_I_new, jnp.array([T_new, n_new]))

            return (ynew, Vnew, t+dt), jnp.append(ynew, jnp.array([Vnew, t+dt]))

        return jax.lax.scan(scanner, (ics, 300.0, 0), jnp.arange(Nt))


    def implicit_euler_step(self, y, Vp, T, n, dt, sheath_solve):
        '''
        Perform a single implicit-Euler step of the RLC circuit equations

        Args:
            y: An array containing [Qn, Qdotn=Ip^n], the 2-vector of unknowns at time t^n
            Vp: The plasma gap voltage in volts at time t^n. Useful as an initial guess for a Newton iteration
            T: The plasma temperature in units of electron-volts
            n: The plasma density in units of meter^-3
            dt: The timestep in units of seconds
            sheath_solve: A Callable that accepts (V, N, T), where
                - V is the plasma gap voltage [volts]
                - T is the plasma temperature [eV]
                - n is the plasma density [m^-3]
                and returns the plasma current in amperes

        Returns: (y, V) where
            y: A 2-vector of [Q, Qdot=Ip] at time t^n+1
            V The plasma gap voltage at time t^n+1
        '''
        
        def residual(Qn1Vn1, args):
            Qn1, Vn1 = Qn1Vn1
            Qn = y[0]
            In = y[1]

            In1 = sheath_solve(Vn1, T, n)

            Vrp1 = (Vn1 + self.Lp / (self.L + self.Lp) * (Qn1/self.C - self.R*In1)) / (1 - self.Lp / (self.L + self.Lp))
            rhs = jnp.array([Qn1 - Qn - dt * In1,
                            -In1 + In + (dt/(self.L-self.Lp)) * (-Qn1/self.C - self.R*In + Vrp1)])
            return rhs
            
        # - A call to optx.root_find that performs the Newton solve with self.rootfinder
        sol = optx.root_find(residual, self.rootfinder, jnp.array([y[0], Vp]))
        Q_new, V_new = sol.value
        I_new = sheath_solve(V_new, T, n)
        return jnp.array([Q_new, I_new]), V_new



    def log_progress(self, t, y, Vp):
        '''
        Log a timestep to the MLFlow server.
        '''
        step_ns = int(t * 1e9)
        Q, I, T, n = y
        mlflow.log_metric("Time - seconds", t, step=step_ns, run_id=self.mlflow_run_id)
        mlflow.log_metric("Capacitor charge - coulombs", Q, step=step_ns, run_id=self.mlflow_run_id)
        mlflow.log_metric("Current - amperes", I, step=step_ns, run_id=self.mlflow_run_id)
        mlflow.log_metric("Temperature - eV", T, step=step_ns, run_id=self.mlflow_run_id)
        mlflow.log_metric("Density - per cubic meter", n, step=step_ns, run_id=self.mlflow_run_id)
        mlflow.log_metric("Voltage - volts", Vp, step=step_ns, run_id=self.mlflow_run_id)


    def step_heating_and_cooling(self, I, T, n, dt):
        '''
        Perform one timestep of the resistive heating and radiative cooling terms.
        These modify only temperature, so we return T_prime, the temperature after heating+cooling,
        but before adiabatic compression or expansion back to thermal equilibrium.

        Returns:
            T_prime, the temperature in units of electron-volts
        '''
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
