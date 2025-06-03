import jax

jax.config.update("jax_enable_x64", True)

from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float64
from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths
import jpu
import jax.numpy as jnp

ureg = jpu.UnitRegistry()

class InputSchema(BaseModel):
    Vp: Differentiable[Float64] = Field(
            description="Anode-cathode voltage gap [volts]"
    )
    T: Differentiable[Float64] = Field(
            description="Plasma temperature [eV]"
    )
    N: Differentiable[Float64] = Field(
            description="Plasma linear density; N_e = N_i = N [1e18 m^-1]"
    )


class OutputSchema(BaseModel):
    Ip: Differentiable[Float64] = Field(
            description="Plasma current [kA]"
    )


def apply(inputs: InputSchema) -> OutputSchema:
    apply_jit(inputs.model_dump())


def apply_jit(inputs: dict) -> dict:
    Vp = inputs["Vp"] * ureg.V
    T = inputs["T"] * ureg.eV
    N = inputs["N"] * 1e18 / ureg.m

    # Compute the saturation current I = 0.5 * e * N * c_S
    gamma = 5/3
    c_S = ((2*gamma*T / ureg.m_p)**0.5).to(ureg.m / ureg.s)
    Ip_sat = 0.5 * ureg.e * N * c_S

    alpha = (Vp * ureg.e / T).to('').magnitude

    Ip = -jnp.tanh(alpha) * Ip_sat

    return dict(Ip=Ip.to(ureg.kA).magnitude)


def vector_jacobian_product(inputs: InputSchema, vjp_inputs: set[str], vjp_outputs: set[str], cotangent_vector: dict):
    return vjp_jit(inputs.model_dump(), tuple(vjp_inputs), tuple(vjp_outputs), cotangent_vector)
            

def vjp_jit(inputs: dict, vjp_inputs: tuple[str], vjp_outputs: tuple[str], cotangent_vector: dict):
    filtered_apply = filter_func(apply_jit, inputs, vjp_outputs)
    _, vjp_func = jax.vjp(filtered_apply, flatten_with_paths(inputs, include_paths=vjp_inputs))
    return vjp_func(cotangent_vector)[0]

