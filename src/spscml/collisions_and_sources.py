import jax.numpy as jnp


def collision_frequency_shape_func(grids, expand_dims=False):
    """
    Calculate collision frequency shape function.
    
    Args:
        grids: Dictionary containing spatial grid with 'x' key
        expand_dims: If True, add extra dimension for broadcasting
        
    Returns:
        Collision frequency shape function
    """
    L = grids['x'].Lx

    midpt = L/4
    # Want 10 e-foldings between the midpoint (2/3rds of the way to the sheath)
    # and the wall
    efolding_dist = (midpt/2)/20
    x = grids['x'].xs
    h0 = lambda x: 1 + jnp.exp((x/efolding_dist) - midpt/efolding_dist)
    h = 1 / (0.5 * (h0(x) + h0(-x)))
    
    if expand_dims:
        return jnp.expand_dims(h, axis=1)
    else:
        return h


def flux_source_shape_func(x_grid):
    """
    Compute flux source shape function for particle injection.
    
    Args:
        x_grid: Spatial grid object
        
    Returns:
        Spatial shape function for flux source
    """
    Ls = x_grid.Lx / 4
    return (1 / Ls - jnp.abs(x_grid.xs) / Ls**2)


def maxwellian(grid, A, n, T=1.0):
    """
    Compute Maxwellian distribution for collision operator.
    
    Args:
        grid: Phase space grid
        A: Mass ratio
        n: Density (can be scalar or array)
        T: Temperature (default 1.0)
        
    Returns:
        Maxwellian distribution in velocity space
    """
    v = grid.vs
    theta = T / A
    M = n / (jnp.sqrt(2*jnp.pi*theta)) * jnp.exp(-v**2 / (2*theta))
    return M
