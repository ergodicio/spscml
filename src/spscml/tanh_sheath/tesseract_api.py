import jax

jax.config.update("jax_enable_x64", True)

from pydantic import BaseModel, Field
from tesseract_runtime import Array, Differentiable, Float64
import jpu

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
    Vp = inputs.Vp * ureg.V
    T = inputs.T * ureg.eV
    N = inputs.N * 1e18 / ureg.m

    # Compute the saturation current I = 0.5 * e * N * c_S
    gamma = 5/3
    c_S = jnp.sqrt(2*gamma*T / ureg.mp).to(ureg.m / ureg.s)
    Ip_sat = 0.5 * ureg.e * N * c_S

    alpha = (Vp * ureg.e / T).to('').magnitude
    Ip = -jnp.tanh(alpha) * Ip_sat
    return Ip.to(ureg.kA).magnitude


#def vector_jacobian_product(inputs: InputSchema, vjp_inputs: set[str], vjp_outputs: set[str], cotangent_vector: dict):
    #return 
