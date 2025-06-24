import jpu
import jax.numpy as jnp
import jax
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract
from functools import partial
import equinox as eqx
import mlflow
import matplotlib.pyplot as plt
import matplotlib as mpl

import sheaths.vlasov.tesseract_api as vlasov_tesseract_api
import sheaths.tanh_sheath.tesseract_api as tanh_sheath_tesseract_api

from .solver import Solver
from ..fusion import fusion_power, bremsstrahlung_power

ureg = jpu.UnitRegistry()


@eqx.filter_jit
def apply_jit(inputs: dict, sheath_tx) -> dict:
    return solve_wdm_with_tesseract(inputs, sheath_tx)


def apply(inputs: dict, sheath_tx) -> dict:
    # Optional: Insert any pre-processing/setup that doesn't require tracing
    # and is only required when specifically running your apply function
    # and not your differentiable endpoints.
    # For example, you might want to set up a logger or mlflow server.
    # Pre-processing should not modify any input that could impact the
    # differentiable outputs in a nonlinear way (a constant shift
    # should be safe)

    with mlflow.start_run(run_name="Whole-device model solve", 
                          parent_run_id=inputs['mlflow_parent_run_id']) as mlflow_run:
        inputs['mlflow_run_id'] = mlflow_run.info.run_id

        for param in ["Vc0", "Ip0", "a0", "N", "Lp_prime", "Lz", 
                      "R", "L", "C"]:
            mlflow.log_param(param, inputs[param])

        out = apply_jit(inputs, sheath_tx)

        mlflow.log_figure(circuit_plots(out), "plots/circuit.png")
        mlflow.log_figure(adiabat_plot(out, inputs['N'], inputs['Lz']), "plots/adiabat.png")
        mlflow.log_metric("TripleProduct", jnp.sum(out["n"] * out["T"] * inputs['dt']))

    # Optional: Insert any post-processing that doesn't require tracing
    # For example, you might want to save to disk or modify a non-differentiable
    # output. Again, do not modify any differentiable output in a non-linear way.
    return out


def solve_wdm_with_tesseract(inputs: dict, sheath_tesseract) -> dict:
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

    def sheath_solve(Vp, T, n):
        tx_inputs = dict(Vp=jnp.array(Vp),
                         n=jnp.array(n),
                         T=jnp.array(T),
                         Lz=jnp.array(Lz),
                         mlflow_parent_run_id=inputs['mlflow_run_id'])
        j = apply_tesseract(sheath_tesseract, tx_inputs)['j']
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
    

## Plotting

def circuit_plots(out):
    ts = out["ts"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(ts * 1e6, out["Ip"] / 1000, label="Plasma current [kA]")
    axes[0].set_ylabel("Plasma current [kA]")
    
    axes[1].plot(ts * 1e6, out["Vp"] / 1000)
    axes[1].set_ylabel("Voltage gap [kV]")
    axes[1].set_xlabel("Time [µs]")
    plt.tight_layout()

    return fig


def adiabat_plot(out, N, L):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    n = out["n"]
    T = out["T"]

    def fusion_power_of(n, T):
        a = jnp.sqrt(N / n / jnp.pi)
        return fusion_power(n, L, a, T)

    Ts = jnp.geomspace(0.7*jnp.min(T), 1.5*jnp.max(T), 100)
    ns = jnp.geomspace(0.7*jnp.min(n), 1.5*jnp.max(n), 100)
    n_mesh, T_mesh = jnp.meshgrid(ns, Ts)
    P_f = fusion_power_of(n_mesh, T_mesh)
    P_f_levels = jnp.geomspace(1e-6, 1e18, 13)
    
    cmap=plt.cm.plasma.copy()
    cmap.set_under('white')
    cf = ax.contourf(n_mesh, T_mesh, P_f, alpha=0.5, levels=P_f_levels,
                     cmap=cmap, norm=mpl.colors.LogNorm(vmin=1e-6, vmax=1e18), 
                     vmin=1e-6, vmax=1e18, extend='both')
    
    cbar = fig.colorbar(cf, extend='both')
    cbar.ax.set_ylabel("Fusion power [watts]")


    ax.loglog(n, T, color='blue')
    Ts = jnp.linspace(0.1*jnp.min(T), 10*jnp.max(T))
    for adiabat in 1.5**jnp.arange(-8, 8):
        ax.loglog((Ts/T[0])**(1.5) * n[0], Ts * adiabat, color='gray', linewidth=1.0)

    ax.set_xlim(0.7*jnp.min(n), 1.5*jnp.max(n))
    ax.set_ylim(0.7*jnp.min(T), 1.5*jnp.max(T))

    ax.set_xlabel("n [m^-3]")
    ax.set_ylabel("T [eV]")

    return fig
