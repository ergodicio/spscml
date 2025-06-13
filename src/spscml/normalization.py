import jpu
import jax.numpy as jnp
import jax

def plasma_norm(T, n):
    """
    Returns a proton plasma period / Debye length normalization for the given temperature
    and density.

    params:
        - T: The plasma temperature in eV
        - n: The density in m^-3
    """
    ureg = jpu.UnitRegistry()

    T0 = T * ureg.eV
    n0 = n * ureg.m**-3
    omega_p_tau = 1.0

    # Use a thermal velocity normalization
    v0 = ((T0 / ureg.m_p)**0.5).to(ureg.m / ureg.s)

    # Proton plasma frequency
    omega_p = ((n0 * ureg.e**2 / (ureg.m_p * ureg.epsilon_0))**0.5).to(ureg.s**-1)
    tau = omega_p_tau / omega_p

    L = v0 * tau
    lambda_D = L

    # Proton-proton collision frequency
    log_lambda = 10

    # Equation (44) of habbershawNonlinearConservativeEntropic2024,
    # evaluated for proton-proton collisions
    xi_proton_proton = 2 / (3 * (2*jnp.pi)**(3/2) * ureg.epsilon_0**2) * log_lambda * ureg.e**4 * n0**2 / (ureg.m_p**2 * (2*T0/ureg.m_p)**(3/2))

    nu_p = (xi_proton_proton / n0).to(ureg.s**-1)
    nu_p_tau = (nu_p * tau).to('').magnitude

    vtp = (T0 / ureg.m_p)**0.5
    # Debye length
    lambda_mfp = (vtp / nu_p).to(ureg.m)

    jax.debug.print("lambda_mfp: {}", lambda_mfp)

    vA = v0
    B0 = ((vA**2 * ureg.m_p * n0 * ureg.mu_0)**0.5).to(ureg.tesla)

    omega_c_tau = (ureg.e * B0 / ureg.m_p * tau).to('').magnitude

    eta_spitzer = (2*ureg.m_e)**0.5 * ureg.e**2 * log_lambda / (1.96*12*jnp.pi**1.5 * ureg.epsilon_0**2 * T0**1.5)

    j0 = v0 * n0 * ureg.e
    E0 = (j0 * eta_spitzer).to(ureg.volt / ureg.meter)
    P_rad = (eta_spitzer * j0**2).to(ureg.eV / ureg.s / ureg.m**3)

    E0 = v0 * B0
    V0 = E0 * L
    j0 = n0 * v0 * ureg.e

    return dict(
        ureg=ureg, T0=T0, n0=n0, v0=v0, L=L, tau=tau, E0=E0, V0=V0, j0=j0,
        omega_p_tau=omega_p_tau, omega_c_tau=omega_c_tau, nu_p_tau=nu_p_tau,
        xi_proton_proton=xi_proton_proton,
        lambda_mfp=lambda_mfp, lambda_D=lambda_D,
    )


