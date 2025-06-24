import jax.numpy as jnp

from .grids import PhaseSpaceGrid

def maxwellian_1d(A, n, nu, T):
    return lambda x, v: n / jnp.sqrt(2*jnp.pi * (T/A)) * jnp.exp(-A*(v-nu/n)**2 / (2*T))

def maxwellian_1d_v2(A, n, nu, T, v):
    return n / jnp.sqrt(2*jnp.pi * (T/A)) * jnp.exp(-A*(v-nu/n)**2 / (2*T))


def zeroth_moment(f, grid: PhaseSpaceGrid):
    return jnp.sum(f, axis=1) * grid.dv


def first_moment(f, grid: PhaseSpaceGrid):
    return jnp.sum(f * grid.vT, axis=1) * grid.dv


def second_moment(f, grid: PhaseSpaceGrid):
    return jnp.sum(f * grid.vT**2 / 2, axis=1) * grid.dv


def temperature(f, A, grid: PhaseSpaceGrid):
    n = zeroth_moment(f, grid)
    u = jnp.expand_dims(first_moment(f, grid) / n, axis=1)
    T = A * jnp.sum(f * (grid.vT - u)**2, axis=1) * grid.dv
    return T
