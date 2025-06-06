import jax

jax.config.update("jax_enable_x64", True)

from typing import Callable
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float64
from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths
from tesseract_jax import apply_tesseract
from tesseract_core import Tesseract
import jpu
import jax.numpy as jnp
import equinox as eqx

from spscml.plasma import TwoSpeciesPlasma
from spscml.rlc_circuit.solver import Solver

ureg = jpu.UnitRegistry()

class InputSchema(BaseModel):
    Vc0: Differentiable[Float64] = Field(
            description="Initial capacitor charge voltage [volts]"
    )
    C: Differentiable[Float64] = Field(
            description="Capacitance [micro-Farads]"
    )
    L: Differentiable[Float64] = Field(
            description="Circuit inductance [nano-Henries]"
    )
    R: Differentiable[Float64] = Field(
            description="Circuit resistance [milli-Ohms]"
    )
    sheath_tesseract_url: str = Field(
            description="The tesseract to use when calculating the plasma current"
    )
    t_end: Float64 = Field(
            description="End time of the simulation [microseconds]"
    )


class OutputSchema(BaseModel):
    Ip_final: Differentiable[Float64] = Field(
            description="Plasma current at final time [amperes]"
    )
    sol_ts: Array[(None,), Float64] = Field(
            description="ODE Solution: timestamps [microseconds]"
    )
    sol_Ip: Array[(None,), Float64] = Field(
            description="ODE Solution: plasma current [amperes]"
    )
    sol_Vp: Array[(None,), Float64] = Field(
            description="ODE Solution: plasma voltage [volts]"
    )
    sol_Q: Array[(None,), Float64] = Field(
            description="ODE Solution: capacitor charge [coulombs]"
    )


def sheath_solve(tx, Vp):
    Ip = apply_tesseract(tx, {"Vp": jnp.array(Vp), 
                              "N": jnp.array(1.0), 
                              "T": jnp.array(9.0e3)})["Ip"]
    return {"Ip": Ip*1000}


def apply(inputs: InputSchema) -> OutputSchema:
    with Tesseract.from_url(inputs.sheath_tesseract_url) as tx:
        return apply_jit(inputs.model_dump(), lambda Vp: sheath_solve(tx, Vp))


@eqx.filter_jit
def apply_jit(inputs: dict, sheath_solve: Callable) -> dict:
    Vc0 = (inputs["Vc0"] * ureg.V).magnitude
    C = (inputs["C"] * ureg.uF).to(ureg.F).magnitude
    L = (inputs["L"] * ureg.nH).to(ureg.H).magnitude
    R = (inputs["R"] * ureg.milliohm).to(ureg.ohm).magnitude

    plasma = TwoSpeciesPlasma(1.0, 1.0, 1.0, 1.0, 0.04, 1.0, -1.0)

    solver = Solver(plasma, R, L, C, -0.1*L, Vc0)

    omega_0 = 1 / jnp.sqrt(C*L) # Units of Hertz
    jax.debug.print("omega_0: {}", omega_0)
    Q0 = C * Vc0 # Coulombs
    I0 = 0.0 # Amperes
    ics = jnp.array([Q0, I0])

    jax.debug.print("Q0: {}", Q0)
    jax.debug.print("R: {}", R)
    jax.debug.print("L: {}", L)
    jax.debug.print("C: {}", C)

    dt = 4e-7

    _, sol = solver.solve(dt=dt, Nt=50, ics=ics, sheath_solve=sheath_solve)

    Q = sol[:, 0]
    Ip = sol[:, 1]
    Vp = sol[:, 2]
    ts = sol[:, 3]

    return dict(Ip_final=Ip[-1], sol_ts=ts, sol_Ip=Ip, sol_Vp=Vp, sol_Q=Q)


