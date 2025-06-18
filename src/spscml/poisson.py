import jax.numpy as jnp

def poisson_solve(grid, plasma, rho_c, boundary_conditions):
    """
    Solves the Poisson equation nabla^2 phi = -omega_c_tau / omega_p_tau^2 * rho_c.

    Returns E = -nabla phi
    """
    oct = plasma.omega_c_tau
    opt = plasma.omega_p_tau
    rhs = -opt**2 / oct * rho_c
    dx = grid.dx

    if boundary_conditions['phi'] == 'periodic':
        n = rho_c.shape[0]
        rhs_hat = jnp.fft.fft(rhs)
        k = jnp.fft.fftfreq(grid.Nx, dx / (2*jnp.pi)).at[0].set(1)
        phi_hat = -rhs_hat / k**2
        phi_hat = phi_hat.at[0].set(0)
        phi = jnp.real(jnp.fft.ifft(phi_hat))
        phi = jnp.concatenate([
            jnp.array([phi[-1]]),
            phi,
            jnp.array([phi[0]])
        ])
        E = -(phi[2:] - phi[:-2]) / (2*dx)
        return E

    
    else:
        left_bc = boundary_conditions['phi']['left']
        right_bc = boundary_conditions['phi']['right']
        left_bc_type = left_bc['type']
        right_bc_type = right_bc['type']

        assert left_bc_type == 'Dirichlet'
        assert right_bc_type == 'Dirichlet'


        # Apply phi boundary conditions
        if left_bc_type == 'Dirichlet':
            rhs = rhs.at[0].add(-left_bc['val'] / dx**2)

        if right_bc_type == 'Dirichlet':
            rhs = rhs.at[-1].add(-right_bc['val'] / dx**2)

        if left_bc_type == 'Dirichlet' and right_bc_type == 'Dirichlet':
            L = grid.laplacian
            phi = jnp.linalg.solve(L, rhs)
            phi = jnp.concatenate([
                jnp.array([left_bc['val']]),
                phi,
                jnp.array([right_bc['val']])
                ])

        E = -(phi[2:] - phi[:-2]) / (2*dx)
        return E

