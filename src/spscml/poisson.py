import jax
import jax.numpy as jnp

def poisson_solve(grid, plasma, rho_c, boundary_conditions):
    """
    Solves the Poisson equation nabla^2 phi = -omega_c_tau / omega_p_tau^2 * rho_c.

    Returns E = -nabla phi
    """
    oct = plasma.omega_c_tau
    opt = plasma.omega_p_tau
    rhs = -opt**2 / oct * rho_c
    
    left_bc = boundary_conditions['phi']['left']
    right_bc = boundary_conditions['phi']['right']
    left_bc_type = left_bc['type']
    right_bc_type = right_bc['type']

    assert left_bc_type == 'Dirichlet'
    assert right_bc_type == 'Dirichlet'

    # Apply phi boundary conditions
    if left_bc_type == 'Dirichlet':
        rhs = rhs.at[0].add(-left_bc['val'] / (grid.cell_to_cell_dxs[0] * grid.face_to_face_dxs[0]))

    if right_bc_type == 'Dirichlet':
        rhs = rhs.at[-1].add(-right_bc['val'] / (grid.cell_to_cell_dxs[-1] * grid.face_to_face_dxs[-1]))

    if left_bc_type == 'Dirichlet' and right_bc_type == 'Dirichlet':
        dl, d, du = grid.laplacian_diagonals
        phi = jax.lax.linalg.tridiagonal_solve(dl, d, du, rhs[:, None]).flatten()
        phi = jnp.concatenate([
            jnp.array([left_bc['val']]),
            phi,
            jnp.array([right_bc['val']])
            ])

    E = -(phi[2:] - phi[:-2]) / (grid.cell_to_cell_dxs[1:] + grid.cell_to_cell_dxs[:-1])
    return E

