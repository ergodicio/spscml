import jpu
import jax.numpy as jnp
import jax
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract
from functools import partial

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

    jax.debug.print("Ip0 = {}", Ip0)
    jax.debug.print("T0 = {}", T0)
    jax.debug.print("n0 = {}", n0)


    ics = jnp.array([Q0, Ip0, T0, n0])

    dt = inputs['dt']
    Nt = int(inputs['t_end'] / dt)

    with Tesseract.from_url(inputs['sheath_tesseract_url']) as tx:
        #@jax.custom_jvp
        def sheath_solve(Vp, T, n):
            tx_inputs = dict(Vp=jnp.array(Vp),
                             n=jnp.array(n),
                             T=jnp.array(T),
                             Lz=Lz)
            j = apply_tesseract(tx, tx_inputs)['j']
            Ip = j * inputs['N'] / n
            jax.debug.print("Vp = {}, N={}, T={}, n={}, Ip = {}", Vp, inputs['N'], T, n, Ip)
            return Ip


        # The argument order is correct: https://docs.jax.dev/en/latest/advanced-autodiff.html#jax-custom-jvp-with-nondiff-argnums
        """
        @sheath_solve.defjvp
        def sheath_solve_jvp(primals, tangents):
            V, T, n = primals
            Vdot, Tdot, ndot = tangents
            # primal_out = sheath_solve(V, T, n)
            # T and V are in comparable units, so h can be relative to V.
            h = V * 1e-7
            Ip1 = sheath_solve(V+h, T, n)
            Ip2 = sheath_solve(V-h, T, n)
            # Save an evaluation by estimating Ip(V) as the average of these two
            primal_out = 0.5*(Ip1 + Ip2)
            tangent_out = (Ip1 - Ip2) / (2*h) * Vdot
            return primal_out, tangent_out
        """


        _, solution = wdm_solver.solve(dt, Nt, ics, sheath_solve)


    Q = solution[:, 0]
    Ip = solution[:, 1]
    T = solution[:, 2]
    n = solution[:, 3]
    Vp = solution[:, 4]
    ts = solution[:, 5]

    return dict(Q=Q, Ip=Ip, T=T, n=n, Vp=Vp, ts=ts)
    


