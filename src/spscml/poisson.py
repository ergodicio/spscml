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

    dx = grid.dx

    # Apply phi boundary conditions
    if left_bc_type == 'Dirichlet':
        rhs = rhs.at[0].add(-left_bc['val'] / dx**2)
    elif left_bc_type == 'Robin':
        robin_coef = -left_bc['alpha'] / dx + left_bc['beta']
        rhs = rhs.at[0].subtract(1 / dx**2 * left_bc['val'] / robin_coef)

    if right_bc_type == 'Dirichlet':
        rhs = rhs.at[-1].add(-right_bc['val'] / dx**2)
    elif right_bc_type == 'Robin':
        robin_coef = right_bc['alpha'] / dx + right_bc['beta']
        rhs = rhs.at[-1].subtract(1 / dx**2 * right_bc['val'] / robin_coef)

    if left_bc_type == 'Dirichlet' and right_bc_type == 'Dirichlet':
        L = grid.laplacian
        phi = jnp.linalg.solve(L, rhs)
        phi = jnp.concatenate([
            jnp.array([left_bc['val']]),
            phi,
            jnp.array([right_bc['val']])
            ])

    elif left_bc_type == 'Robin' and right_bc_type == 'Dirichlet':
        L = grid.robin_dirichlet_laplacian(left_bc['alpha'], left_bc['beta'])
        phi = jnp.linalg.solve(L, rhs)
        # beta*phiL + alpha*(phi[0]-phiL)/dx = robin_val
        coef = left_bc['beta'] - left_bc['alpha'] / dx
        phiL = (left_bc['val'] - phi[0] * left_bc['alpha'] / dx) / coef
        phi = jnp.concatenate([
            jnp.array([phiL]),
            phi,
            jnp.array([right_bc['val']]) 
            ])

    elif left_bc_type == 'Dirichlet' and right_bc_type == 'Robin':
        L = grid.dirichlet_robin_laplacian(right_bc['alpha'], right_bc['beta'])
        phi = jnp.linalg.solve(L, rhs)
        phiR = (right_bc['val'] + phi[-1] * right_bc['alpha'] / dx) / robin_coef
        phi = jnp.concatenate([
            jnp.array([left_bc['val']]),
            phi,
            jnp.array([phiR])
            ])

    E = -(phi[2:] - phi[:-2]) / (2*dx)
    return E

