import jpu
import jax.numpy as jnp

ureg = jpu.UnitRegistry()

def fusion_power(n, L, a, T):
    """
    params:
        - n: The volumetric density [m^-3]
        - L: The pinch length [m]
        - a: The pinch radius [m]
        - T: The average plasma temperature [eV]
    """

    T_keV = (T * ureg.eV).to(ureg.keV).magnitude
    sigma_nu_DT = 3.67e-18 * T_keV**(-2/3) * jnp.exp(-19.94*T_keV**(-1/3)) * ureg.m**3 / ureg.s
    a = a * ureg.m
    L = L * ureg.m
    n = n * ureg.m**-3
    P_f = (n**2 / 4 * sigma_nu_DT * (17.6*ureg.MeV) * jnp.pi * a**2 * L)
    return P_f.to(ureg.W).magnitude


# Returns the Bremsstrahlung power in units of eV / s
def bremsstrahlung_power(n, L, a, T):
    P_br = n**2 * T**0.5 / (7.69e18)**2 * ureg.W / ureg.m**-3

    L = L * ureg.m
    a = a * ureg.m
    vol = L * jnp.pi*a**2
    P_rad = (P_br * vol).to(ureg.W)
    return P_rad
