import sys
sys.path.append("src")

import jax.numpy as jnp
import jax

from spscml.fulltensor_vfp.sheath_model import calculate_plasma_current

jax.config.update("jax_enable_x64", True)

def estimate_resistive_plasma_voltage(Lp, Qdot):
    V_Rp_guess = 1.0

    def residual(V_Rp):
        V_total = V_Rp + Lp * Qdot
        Ip = calculate_plasma_current(V_total)
        jax.debug.print("Plasma current: {}", Ip)
        return Ip - Qdot

    vg = jax.value_and_grad(residual)
    
    for i in range(10):
        r_actual, r_grad = vg(V_Rp_guess)
        jax.debug.print("r actual: {}", r_actual)
        jax.debug.print("grad: {}", r_grad)
        step = -r_actual / r_grad
        step = jnp.sign(step) * jnp.minimum(jnp.abs(step), 0.5)
        V_Rp_guess = V_Rp_guess + step
        jax.debug.print("guess: {}", V_Rp_guess)

    return V_Rp_guess


estimate_resistive_plasma_voltage(0.0, -0.7)
