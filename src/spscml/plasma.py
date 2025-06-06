import equinox as eqx


class TwoSpeciesPlasma(eqx.Module):
    """
    Encapsulates the normalized parameters defining a two-species plasma
    of ions and electrons.
    """
    omega_p_tau: float
    omega_c_tau: float
    nu_p_tau: float
    Ai: float
    Ae: float
    Zi: float
    Ze: float


