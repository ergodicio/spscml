import equinox as eqx
import numpy as np
import jax.numpy as jnp
from jax import Array
import jax

class PhaseSpaceGrid(eqx.Module):
    # x information
    Lx: float
    Nx: int
    face_locs: Array
    cell_centers: Array
    cell_widths: Array
    face_to_face_dxs: Array
    cell_to_cell_dxs: Array
    laplacian_diagonals: tuple[Array]

    # velocity information
    vmax: float
    Nv: int
    dv: float
    vs: Array
    vT: Array
    v_faces_T: Array

    xv: tuple[Array]

    def __init__(self, Lx, Nx, face_locs, cell_centers, cell_widths, cell_to_cell_dxs, laplacian_diagonals, 
                 vmax, Nv):
        self.Lx = Lx
        self.vmax = vmax
        self.Nx = Nx
        self.Nv = Nv

        self.face_locs = face_locs
        self.cell_centers = cell_centers
        self.cell_widths = cell_widths
        self.face_to_face_dxs = cell_widths
        self.cell_to_cell_dxs = cell_to_cell_dxs
        self.laplacian_diagonals = laplacian_diagonals

        dv = 2*vmax / Nv
        self.dv = dv

        self.vs = jnp.linspace(-vmax + dv/2, vmax - dv/2, Nv)

        self.vT = jnp.atleast_2d(self.vs)
        self.v_faces_T = jnp.atleast_2d(jnp.linspace(-vmax, vmax, Nv+1))

        self.xv = jnp.meshgrid(self.cell_centers, self.vs, indexing='ij')


class Grid():
    Lx: float
    Nx: int
    face_locs: Array
    cell_centers: Array
    cell_widths: Array
    cell_to_cell_dxs: Array
    face_to_face_dxs: Array
    laplacian_diagonals: tuple[Array]

    def __init__(self, face_locs):
        self.Lx = face_locs[-1] - face_locs[0]
        self.Nx = face_locs.shape[0] - 1

        self.face_locs = face_locs
        self.cell_centers = 0.5*(face_locs[1:] + face_locs[:-1])
        self.cell_widths = jnp.diff(face_locs)
        self.face_to_face_dxs = self.cell_widths
        print(self.face_to_face_dxs.shape)
        cell_to_cell_dxs = jnp.diff(self.cell_centers)
        self.cell_to_cell_dxs = jnp.concatenate([
            jnp.array([cell_to_cell_dxs[0]]),
            cell_to_cell_dxs,
            jnp.array([cell_to_cell_dxs[-1]]),
        ])
        print(self.cell_to_cell_dxs.shape)

        self.laplacian_diagonals = laplacian_nonuniform_diagonals(self.cell_to_cell_dxs, self.cell_widths)


    def extend_to_phase_space(self, vmax, Nv):
        return PhaseSpaceGrid(self.Lx, self.Nx, self.face_locs, self.cell_centers, self.cell_widths, 
                              self.cell_to_cell_dxs, self.laplacian_diagonals, vmax, Nv)


def laplacian_nonuniform_diagonals(cell_to_cell_dxs, face_to_face_dxs):
    l = 2 / (cell_to_cell_dxs[:-1] * (cell_to_cell_dxs[:-1] + cell_to_cell_dxs[1:]))
    c = -2 / (cell_to_cell_dxs[:-1] * cell_to_cell_dxs[1:])
    r = 2 / (cell_to_cell_dxs[1:] * (cell_to_cell_dxs[:-1] + cell_to_cell_dxs[1:]))
    dl = jnp.append(jnp.array([0.]), l[1:])
    d = c
    du = jnp.append(r[:-1], jnp.array([0.]))

    return (dl, d, du)

