import equinox as eqx
import numpy as np
import jax.numpy as jnp


class PhaseSpaceGrid():
    def __init__(self, Lx, vmax, Nx, Nv):
        self.Lx = Lx
        self.vmax = vmax
        self.Nx = Nx
        self.Nv = Nv

        dx = Lx / Nx
        self.dx = dx
        self.dv = 2*vmax / Nv

        self.xs = jnp.linspace(-Lx/2 + dx/2, Lx/2 - dx/2, Nx)
        self.vs = jnp.linspace(-vmax + dv/2, vmax - dv/2, Nv)

        self.xv = jnp.meshgrid(xs, vs, indexing='ij')

        self.laplacian = laplacian(Nx, dx)


class Grid():
    def __init__(self, Nx, Lx):
        self.Lx = Lx
        self.Nx = Nx

        dx = Lx / Nx
        self.dx = dx

        self.xs = jnp.linspace(-Lx/2 + dx/2, Lx/2 - dx/2, Nx)
        self.laplacian = laplacian(Nx, dx)


def laplacian(Nx, dx):
    L = np.zeros((Nx, Nx))
    for i in range(Nx):
        L[i, i] = -2
        if i > 0:
            L[i-1, i] = 1
            L[i, i-1] = 1

    L = jnp.array(L) / (self.dx**2)
    return L
