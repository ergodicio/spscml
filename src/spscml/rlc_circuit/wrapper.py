import jpu
import jax.numpy as jnp
import jax
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

from .solver import Solver

ureg = jpu.UnitRegistry()

def solve_wdm(inputs: dict) -> dict:
    Lp_prime = inputs['Lp_prime']
    Lz = inputs['Lz']
    Lp = Lp_prime * Lz
    wdm_solver = Solver(
        R=inputs['R'], L=inputs['L'], C=inputs['C'], Lp=Lp,
        V0=inputs['Vc0'], Lz=Lz, N=inputs['N'])

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
                             N=jnp.array(inputs['N']),
                             n=jnp.array(n),
                             T=jnp.array(T))
            return apply_tesseract(tx, tx_inputs)['Ip']

        _, solution = wdm_solver.solve(dt, Nt, ics, sheath_solve)


    Q = solution[:, 0]
    Ip = solution[:, 1]
    T = solution[:, 2]
    n = solution[:, 3]
    Vp = solution[:, 4]
    ts = solution[:, 5]

    return dict(Q=Q, Ip=Ip, T=T, n=n, Vp=Vp, ts=ts)
    


