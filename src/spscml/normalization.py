import jpu
import jax.numpy as jnp

def plasma_norm():
    ureg = jpu.UnitRegistry()

    T0 = 2.0 * ureg.keV
    n0 = 1.1e24 * (ureg.m**-3)
    N0 = 1e18 * (ureg.m**-1)

    v0 = ((T0 / ureg.m_p)**0.5).to(ureg.m / ureg.s)

    # N0 = n0*L^2 -> L = sqrt(N0 / n0)
    L = (N0 / n0)**0.5
    tau = (L / v0).to(ureg.s)

    # Proton plasma frequency
    omega_p = ((n0 * ureg.e**2 / (ureg.m_p * ureg.epsilon_0))**0.5).to(ureg.s**-1)
    omega_p_tau = (omega_p * tau).to('').magnitude

    # Proton-proton collision frequency
    log_lambda = 10
    nu_p = ureg.e**4 * n0 * log_lambda / (3*jnp.sqrt(2*jnp.pi**3) * ureg.epsilon_0**2 * ureg.m_p**0.5 * T0**1.5)
    nu_p = nu_p.to(ureg.s**-1)
    nu_p_tau = (nu_p * tau).to('').magnitude

    vtp = (T0 / ureg.m_p)**0.5
    # Debye length
    lambda_D = (vtp / omega_p).to(ureg.m)
    lambda_mfp = (vtp / nu_p).to(ureg.m)

    vA = v0
    B0 = ((vA**2 * ureg.m_p * n0 * ureg.mu_0)**0.5).to(ureg.tesla)

    omega_c_tau = (ureg.e * B0 / ureg.m_p * tau).to('').magnitude

    print((50*ureg.cm / lambda_D).to(''))
    print(lambda_mfp)
    print(omega_c_tau)

    eta_spitzer = (2*ureg.m_e)**0.5 * ureg.e**2 * log_lambda / (1.96*12*jnp.pi**1.5 * ureg.epsilon_0**2 * T0**1.5)

    j0 = v0 * n0 * ureg.e
    E0 = (j0 * eta_spitzer).to(ureg.volt / ureg.meter)
    P_rad = (eta_spitzer * j0**2).to(ureg.eV / ureg.s / ureg.m**3)

    return dict(
        T0=T0, n0=n0, N0=N0, v0=v0, L=L, tau=tau, omega_p_tau=omega_p_tau, omega_c_tau=omega_c_tau,
        nu_p_tau=nu_p_tau, lambda_mfp=lambda_mfp
    )


