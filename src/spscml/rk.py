import jax
import jax.numpy as jnp

def rk1(y, rhs, dt):
    """
    The first-order Forward Euler method
    """
    return jax.tree.map(lambda u, du: u + dt*du, y, rhs(y))


def ssprk2(y, rhs, dt):
    """
    The usual SSPRK2 second-order method
    """
    y_star = jax.tree.map(lambda u, du: u + dt*du, y, rhs(y))
    y_next = jax.tree.map(lambda u, u_star, du: u/2 + 0.5*(u_star + dt*du),
                          y, y_star, rhs(y_star))
    return y_next


def imex_euler(yn, nonstiff_rhs, stiff_rhs, stiff_implicit_solver, dt):
    rhs1 = jax.tree.map(lambda yn, ns_rhs: yn + dt * ns_rhs, yn, nonstiff_rhs(yn))
    return stiff_implicit_solver(rhs1, dt)


def imex_ssp2(yn, nonstiff_rhs, stiff_rhs, stiff_implicit_solver, dt):
    gamma = 1 - 1 / jnp.sqrt(2)

    y1 = stiff_implicit_solver(yn, dt*gamma)
    rhs1 = jax.tree.map(lambda yn, ns_rhs_1, s_rhs_1: yn + dt*ns_rhs_1 + dt*(1 - 2*gamma)*s_rhs_1,
                        yn, nonstiff_rhs(y1), stiff_rhs(y1))

    y2 = stiff_implicit_solver(rhs1, dt*gamma)

    #y_next = yn + 0.5 * dt * (nonstiff_rhs(y1) + nonstiff_rhs(y2) + stiff_rhs(y1) + stiff_rhs(y2))
    y_next = jax.tree.map(lambda yn, n_rhs_1, n_rhs_2, s_rhs_1, s_rhs_2: yn + 0.5*dt*(n_rhs_1 + n_rhs_2 + s_rhs_1 + s_rhs_2),
                          yn, nonstiff_rhs(y1), nonstiff_rhs(y2), stiff_rhs(y1), stiff_rhs(y2))
    return y_next

