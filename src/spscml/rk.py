def ssprk2(y, rhs, dt):
    """
    The usual SSPRK2 second-order method
    """
    y_star = y + dt*rhs(y)
    y_next = y/2 + 0.5*(y_star + dt*rhs(y_star))
    return y_next
