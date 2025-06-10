import jax
import jax.numpy as jnp


def minmod2(a, b):
    return jnp.where(jnp.abs(a) < jnp.abs(b), a, b)


def minmod3(a, b, c):
    minabs = jnp.minimum(jnp.minimum(jnp.abs(a), jnp.abs(b)), jnp.abs(c))
    return jnp.where(jnp.abs(a) == minabs,
              a,
              jnp.where(jnp.abs(b) == minabs,
                        b,
                        c))


# Port of Script 10.10 from Hesthaven "Numerical Methods for Conservation Laws", 2018
def slope_limiter(a, b, limiter_type):
    if limiter_type == 'minmod':
        return minmod2(a, b)
    elif limiter_type == 'MUSCL':
        return minmod3(0.5*(a+b), 2*a, 2*b)
    elif limiter_type == 'vanLeer':
        return minmod3(2*a*b/(a+b), 2*a, 2*b)
    else:
        raise ValueError(f"Unknown limiter type `{limiter_type}`")


def slope_limited_face_values(cell_averages, limiter_type, axis=0):
    """
    Compute the slope-limited approximation to a grid function at the left
    and right side of each face.

    Accepts an array `cell_averages` of size N+4 along the given axis,
    since we assume that boundary conditions have been applied already.

    Returns a dictionary with shape {'left', 'right'}, each of which values have
    size N+2 along the given axis, corresponding to the N+2 cells in the domain 
    plus one layer of ghost cells.

    params:
        cell_averages: The cell-averaged values of the grid function
        slope_limiter: The type of slope limiter to use.
        axis: The array axis along which to compute differences
    """

    left_slices = [slice(None)]*cell_averages.ndim
    left_slices[axis] = slice(None, -1)

    right_slices = [slice(None)]*cell_averages.ndim
    right_slices[axis] = slice(1, None)

    differences = cell_averages[*right_slices] - cell_averages[*left_slices]

    limited_differences = slope_limiter(differences[*left_slices], 
                                        differences[*right_slices], 
                                        limiter_type)

    interior_slices = [slice(None)]*cell_averages.ndim
    interior_slices[axis] = slice(1, -1)
    interior_values = cell_averages[*interior_slices]

    # The value on the left and right side of each cell
    left_values = interior_values - limited_differences / 2
    right_values = interior_values + limited_differences / 2

    return dict(left=left_values, right=right_values)


def slope_limited_flux_divergence(cell_averages, slope_limiter, numerical_flux, dx, axis=0):
    """
    Computes the flux divergence of numerical_flux applied to the array of cell_averages.

    `numerical_flux` is called like `numerical_flux(left_val, right_val)`, where the two
    values are the piecewise linear approximation to the solution at the left and right
    of the face.
    """
    face_vals = slope_limited_face_values(cell_averages, slope_limiter, axis=axis)
    left_vals = face_vals['left']
    right_vals = face_vals['right']

    if axis == 0:
        right_face_vals = left_vals[1:, ...]
        left_face_vals = right_vals[:-1, ...]
    elif axis == 1:
        right_face_vals = left_vals[:, 1:, ...]
        left_face_vals = right_vals[:, :-1, ...]

    F = numerical_flux(left_face_vals, right_face_vals)
    return jnp.diff(F, axis=axis) / dx
