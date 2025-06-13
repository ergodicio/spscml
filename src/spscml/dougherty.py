from ..utils import zeroth_moment, first_moment, temperature


def species_info(f, A, Z, grid):
    n = zeroth_moment(f, grid)
    nu = first_moment(f, grid)
    u = nu / n
    T = temperature(f, A, grid)
    return dict(f=f, A=A, Z=Z, grid=grid,
                n=n, u=u, T=T)


def lbo_operator_ij(species_i, species_j, norm):
    mixture_moments = lbo_mixture_moments(species_i, species_j, norm)
    fi = species_i["f"]
    fj = species_j["f"]

    grid = species_i["grid"]
    ni = jnp.expand_dims(species_i["n"], axis=1)
    theta_ij = jnp.expand_dims(mixture_moments["T"], axis=1) / species_i["A"]
    u_ij = jnp.expand_dims(mixture_moments["u"], axis=1)

    jax.debug.print(species_i["f"])

    M_ij = ni / (2*jnp.pi*theta_ij) * jnp.exp(-(grid.vT - u_ij)**2 / (2*theta_ij))

    # Apply Dirichlet BCs
    M_ij_bcs = jnp.concatenate([
        -M_ij[:, 0],
        M_ij, 
        -M_ij[:, -1]
    ], axis=1)

    # Equations (73)
    a_ks = (M_ij_bcs[:, :-2] + M_ij_bcs[:, 1:-1]) / (2*M_ij_bcs[:, :-2])
    b_ks = (M_ij_bcs[:, :-2] + 2*M_ij_bcs[:, 1:-1] + M_ij_bcs[:, 2:]) / (2*M_ij_bcs[:, 1:-1])
    c_ks = (M_ij_bcs[:, 1:-1] + M_ij_bcs[:, 2:]) / (2*M_ij_bcs[:, 2:])

    fi_bcs = jnp.concatenate([
        jnp.zeros(grid.Nx),
        fi
        jnp.zeros(grid.Nx),
    ], axis=1)

    L_ks = (a_ks * fi[:, :-2] - b_ks * fi[:, 1:-1] + c_ks * fi[:, 2:]) / (grid.dv**2)

    lambda_ij = mixture_moments["lambda"]

    return lambda_ij * theta_ij * L_ks


def lbo_mixture_moments(species_i, species_j, norm):
    Ai = species_i["A"]
    Zi = species_i["Z"]
    ni = species_i["n"]
    Ti = species_i["T"]

    Aj = species_j["A"]
    Zj = species_j["Z"]
    nj = species_j["n"]
    Tj = species_j["T"]

    params_ij = lbo_parameters(species_i, species_j, norm)
    params_ji = lbo_parameters(species_j, species_i, norm)

    alpha_ij = params_ij["alpha"]
    beta_ij = params_ij["beta"]
    gamma_ij = params_ij["gamma"]
    lambda_ij = params_ij["lambda"]

    alpha_ji = params_ji["alpha"]
    beta_ji = params_ji["beta"]
    gamma_ji = params_ji["gamma"]
    lambda_ji = params_ji["lambda"]

    delta_ij = ni*Ai*lambda_ij * (1 - alpha_ij)

    u_ij = alpha_ij * ui + alpha_ji * uj
    T_ij = beta_ij*Ti + beta_ji*Tj + delta_ij * (ui - uj)**2 / (lambda_ij*ji + lambda_ji*nj)

    return {"lambda": lambda_ij, "u": u_ij, "T": T_ij}



def lbo_parameters(species_i, species_j, norm):
    # Normalized units
    Ai = species_i["A"]
    Zi = species_i["Z"]
    ni = species_i["n"]
    Ti = species_i["T"]

    Aj = species_j["A"]
    Zj = species_j["Z"]
    nj = species_j["n"]
    Tj = species_j["T"]

    xi_ij = norm["xi_proton_proton"] * (Zi*Zj)**2 * ni*nj / (Ai*Aj*(Ti/Ai + Tj/Aj)**(3/2))

    mu_ij = (Ai + Aj) / (2*Ai)
    kappa_ij = 2*mu_ij

    lambda_ij = kappa_ij * xi_ij / ni

    # (48a)
    alpha_ij = 1 - 0.5*(Ai+Aj)/Ai * xi_ij / (ni * lambda_ij)
    # (48b)
    beta_ij = 1 - 0.5*xi_ij/(ni * lambda_ij)
    # (48c)
    gamma_ij = Ai*Aj/(Ai+Aj) * (1-alpha_ij)

    return {"alpha": alpha_ij, "beta": beta_ij, "gamma": gamma_ij, "lambda": lambda_ij}
