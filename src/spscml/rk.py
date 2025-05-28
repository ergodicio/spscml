import jax

def ssprk2(y, rhs, dt):
    """
    The usual SSPRK2 second-order method
    """
    y_star = jax.tree.map(lambda u, du: u + dt*du, y, rhs(y))
    y_next = jax.tree.map(lambda u, u_star, du: u/2 + 0.5*(u_star + dt*du),
                          y, y_star, rhs(y_star))
    return y_next
