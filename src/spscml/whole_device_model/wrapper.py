import jpu
import jax.numpy as jnp
import jax
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract
from functools import partial

from .solver import Solver

ureg = jpu.UnitRegistry()

def solve_wdm(inputs: dict) -> dict:
    '''
    Solve the whole-device model given a dictionary of inputs

    inputs:
        - 'Lz': The interelectrode gap [meters]
        - 'R': The circuit resistance [ohms]
        - 'L': The circuit inductance [henries]
        - 'C': The circuit capacitance [farads]
        - 'Lp_prime': The plasma inductance per unit length [henries/meter]
        - 'Vc0': The initial capacitor voltage
        - 'Ip0': The initial plasma current [amperes]
        - 'a0': The initial pinch radius [meters]
        - 'N': The plasma linear density [meters^-1]
        - 'dt': The timestep to take [seconds]
        - 't_end': The final time to step to [seconds]
        - 'sheath_tesseract_url': The url of a running tesseract which implements the sheath interface.
          See sheath_interface.py

    Returns: a dictionary containing the solution component timeseries
        - Q: the capacitor charge [coulombs]
        - Ip: the plasma current [amperes]
        - T: the plasma temperature [eV]
        - n: the plasma density [m^-3]
        - Vp: the plasma gap voltage [volts]
        - ts: the timesteps taken [seconds]
    '''

    Lp_prime = inputs['Lp_prime']
    Lz = inputs['Lz']
    Lp = Lp_prime * Lz
    wdm_solver = Solver(
        R=inputs['R'], L=inputs['L'], C=inputs['C'], Lp=Lp,
        V0=inputs['Vc0'], Lz=Lz, N=inputs['N'],
        mlflow_run_id=inputs['mlflow_run_id'])

    ## Initial conditions
    Ip0 = inputs['Ip0']
    # Solve the Bennett relation for initial temperature
    P0 = ureg.mu_0 * (Ip0 * ureg.ampere)**2 / (8*jnp.pi)
    T0 = (P0 / (2 * inputs['N'] * ureg.m**-1)).to(ureg.eV).magnitude
    n0 = inputs['N'] / (jnp.pi * inputs['a0']**2)
    Q0 = inputs['Vc0'] * inputs['C']

    ics = jnp.array([Q0, Ip0, T0, n0])

    dt = inputs['dt']
    Nt = int(inputs['t_end'] / dt)

    with Tesseract.from_url(inputs['sheath_tesseract_url']) as tx:
        def sheath_solve(Vp, T, n):
            tx_inputs = dict(Vp=jnp.array(Vp),
                             n=jnp.array(n),
                             T=jnp.array(T),
                             Lz=Lz,
                             mlflow_parent_run_id=inputs['mlflow_run_id'])
            j = apply_tesseract(tx, tx_inputs)['j']
            Ip = j * inputs['N'] / n
            return Ip

        _, solution = wdm_solver.solve(dt, Nt, ics, sheath_solve)


    Q = solution[:, 0]
    Ip = solution[:, 1]
    T = solution[:, 2]
    n = solution[:, 3]
    Vp = solution[:, 4]
    ts = solution[:, 5]

    return dict(Q=Q, Ip=Ip, T=T, n=n, Vp=Vp, ts=ts)
    


