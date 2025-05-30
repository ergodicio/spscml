import jax.numpy as jnp

from .grids import PhaseSpaceGrid

def maxwellian_1d(A, n, nu, T):
    return lambda x, v: n / jnp.sqrt(2*jnp.pi * (T/A)) * jnp.exp(-A*(v-nu/n)**2 / (2*T))


def zeroth_moment(f, grid: PhaseSpaceGrid):
    return jnp.sum(f, axis=1) * grid.dv


def first_moment(f, grid: PhaseSpaceGrid):
    return jnp.sum(f * grid.vT, axis=1) * grid.dv
