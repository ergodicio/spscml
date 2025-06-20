import jax

jax.config.update("jax_enable_x64", True)

from typing import Any
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float64
from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths
import jpu
import jax.numpy as jnp
import equinox as eqx

from spscml.sheath_interface import SheathInputSchema, SheathOutputSchema

ureg = jpu.UnitRegistry()


class InputSchema(SheathInputSchema):
    pass


class OutputSchema(SheathOutputSchema):
    pass


def apply(inputs: InputSchema) -> OutputSchema:
    return apply_jit(inputs.model_dump())


@jax.jit
def apply_jit(inputs: dict) -> dict:
    Vp = inputs["Vp"] * ureg.V
    T = inputs["T"] * ureg.eV
    n = inputs["n"] * (ureg.m**-3)

    # Compute the saturation current I = 0.5 * e * n * c_S
    gamma = 5/3
    c_S = ((2*gamma*T / ureg.m_p)**0.5).to(ureg.m / ureg.s)
    j_sat = 0.5 * ureg.e * n * c_S

    alpha = (Vp * ureg.e / T).to('').magnitude
    log_lambda = 10
    eta_spitzer = (2*ureg.m_e)**0.5 * ureg.e**2 * log_lambda / (1.96*12*jnp.pi**1.5 * ureg.epsilon_0**2 * T**1.5)
    E = Vp / (inputs["Lz"]*ureg.m)

    j_eta = -(E / eta_spitzer).to(ureg.A / ureg.m**2)
    j_sheath = (-jnp.tanh(alpha) * j_sat)
    # Values are negative, so take the maximum
    j = jnp.maximum(j_eta.magnitude, j_sheath.magnitude)

    result = dict(j=j)
    return result


def jacobian(
    inputs: InputSchema,
    jac_inputs: set[str],
    jac_outputs: set[str],
):
    return jac_jit(inputs.model_dump(), tuple(jac_inputs), tuple(jac_outputs))


@eqx.filter_jit
def jac_jit(
    inputs: dict,
    jac_inputs: tuple[str],
    jac_outputs: tuple[str],
):
    filtered_apply = filter_func(apply_jit, inputs, jac_outputs)
    return jax.jacrev(filtered_apply)(
        flatten_with_paths(inputs, include_paths=jac_inputs)
    )



def vector_jacobian_product(inputs: InputSchema, vjp_inputs: set[str], vjp_outputs: set[str], cotangent_vector: dict):
    return vjp_jit(inputs.model_dump(), tuple(vjp_inputs), tuple(vjp_outputs), cotangent_vector)
            

@eqx.filter_jit
def vjp_jit(inputs: dict, vjp_inputs: tuple[str], vjp_outputs: tuple[str], cotangent_vector: dict):
    filtered_apply = filter_func(apply_jit, inputs, vjp_outputs)
    _, vjp_func = jax.vjp(filtered_apply, flatten_with_paths(inputs, include_paths=vjp_inputs))
    return vjp_func(cotangent_vector)[0]


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
):
    return jvp_jit(
        inputs.model_dump(),
        tuple(jvp_inputs),
        tuple(jvp_outputs),
        tangent_vector,
    )


@eqx.filter_jit
def jvp_jit(
    inputs: dict, jvp_inputs: tuple[str], jvp_outputs: tuple[str], tangent_vector: dict
):
    filtered_apply = filter_func(apply_jit, inputs, jvp_outputs)
    return jax.jvp(
        filtered_apply,
        [flatten_with_paths(inputs, include_paths=jvp_inputs)],
        [tangent_vector],
    )[1]


def abstract_eval(abstract_inputs):
    """Calculate output shape of apply from the shape of its inputs."""
    is_shapedtype_dict = lambda x: type(x) is dict and (x.keys() == {"shape", "dtype"})
    is_shapedtype_struct = lambda x: isinstance(x, jax.ShapeDtypeStruct)

    jaxified_inputs = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(**x) if is_shapedtype_dict(x) else x,
        abstract_inputs.model_dump(),
        is_leaf=is_shapedtype_dict,
    )
    dynamic_inputs, static_inputs = eqx.partition(
        jaxified_inputs, filter_spec=is_shapedtype_struct
    )

    def wrapped_apply(dynamic_inputs):
        inputs = eqx.combine(static_inputs, dynamic_inputs)
        return apply_jit(inputs)

    jax_shapes = jax.eval_shape(wrapped_apply, dynamic_inputs)
    return jax.tree.map(
        lambda x: (
            {"shape": x.shape, "dtype": str(x.dtype)} if is_shapedtype_struct(x) else x
        ),
        jax_shapes,
        is_leaf=is_shapedtype_struct,
    )
