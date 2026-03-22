from functools import partial
import json
import time

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap, grad, jacobian

# Specify double precision floats before importing SBPLite4py
jax.config.update("jax_enable_x64", True)

from sbplite4py.dissipation import dissipation_local_lax_friedrichs
from sbplite4py.mesh import UniformMesh1D
from sbplite4py.ref_elem import LegendreGaussLobatto1D
from sbplite4py.flux_difference import evaluate_weak_flux_derivative_1d
from sbplite4py.equations.eulerquasi1d import EulerQuasi1D, flux_chan_et_al
from sbplite4py.utils.odeint import LSERK4
from sbplite4py.utils.errors import get_least_squares_rate  # noqa: E402


def rho_exact(x, t):
    return (
        1 + 0.1 * jnp.sin(2 * jnp.pi * x) + 0.1 * jnp.cos(2 * jnp.pi * x)
    ) * jnp.exp(-t)


def v1_exact(x, t):
    return (
        1 + 0.1 * jnp.sin(2 * jnp.pi * x) + 0.1 * jnp.cos(2 * jnp.pi * x)
    ) * jnp.exp(-t)


def p_exact(x, t):
    return (
        1 + 0.1 * jnp.sin(2 * jnp.pi * x) + 0.1 * jnp.cos(2 * jnp.pi * x)
    ) * jnp.exp(-t)


def area_func(x):
    return 1 - 0.1 * (1 + jnp.cos(2 * jnp.pi * (x - 0.5)))


xl = 0.0
xr = 1.0
equation = EulerQuasi1D(gamma=1.4)


def u_exact(x, t):
    rho = rho_exact(x, t)
    v1 = v1_exact(x, t)
    p = p_exact(x, t)
    a = area_func(x)
    return equation.primitive_to_conserved(rho, v1, p, a)


def manufactured_solution_source_term(x, t):
    p = p_exact(x, t)
    ax = grad(area_func)(x)
    fx = jacobian(lambda x, t: equation.flux(u_exact(x, t)), 0)(x, t)
    ut = jacobian(u_exact, 1)(x, t)
    return ut + fx - jnp.array([0, ax * p, 0, 0])


def initial_condition(x):
    return u_exact(x, 0)


@partial(jit, static_argnums=2)
def rhs(t, u, args: tuple[UniformMesh1D]):
    (mesh,) = args
    uf_interior = mesh.get_internal_face_state(u)
    uf_exterior = mesh.get_external_face_state(uf_interior)

    # Impose inflow BC
    rho_inflow = rho_exact(xl, t)
    v1_inflow = v1_exact(xl, t)
    p_inflow = p_exact(xl, t)
    u_inflow = equation.primitive_to_conserved(
        rho_inflow, v1_inflow, p_inflow, area_func(xl)
    )
    uf_exterior = uf_exterior.at[0, 0].set(u_inflow)

    # Impose outflow BC
    rho_outflow = rho_exact(xr, t)
    v1_outflow = v1_exact(xr, t)
    p_outflow = p_exact(xr, t)
    u_outflow = equation.primitive_to_conserved(
        rho_outflow, v1_outflow, p_outflow, area_func(xr)
    )
    uf_exterior = uf_exterior.at[-1, -1].set(u_outflow)

    # Calculuate surface contributions
    interface_flux = flux_chan_et_al(uf_interior, uf_exterior, equation)
    lxf_penalty = dissipation_local_lax_friedrichs(uf_interior, uf_exterior, equation)
    normals = np.expand_dims(mesh.normals, axis=-1)  # broadcast over each PDE
    mj = jnp.expand_dims(mesh.mj, axis=-1)  # broadcast over each PDE
    surface_contribs = -(1 / mj) * jnp.einsum(
        "ij,kjs->kis", mesh.ref_elem.LIFT, normals * interface_flux + lxf_penalty
    )

    # Calculuate volume contributions
    P_inv = 1 / (mesh.mj * mesh.ref_elem.P)[..., None]
    fx = P_inv * evaluate_weak_flux_derivative_1d(u, mesh.ref_elem, equation)
    volume_contribs = -fx

    # Compute source term for manufactured solution
    source = vmap(vmap(manufactured_solution_source_term, (0, None)), (0, None))(
        mesh.x, t
    )  # (K, Np, 4)

    du = volume_contribs + surface_contribs + source
    return du


def solve_and_compute_error(K, p):
    K = int(K)
    p = int(p)

    num_elements = K
    degree = p
    num_nodes = degree + 1
    ref_elem = LegendreGaussLobatto1D(degree)
    ref_elem.LIFT = jnp.diag(1 / ref_elem.P) @ jnp.eye(num_nodes)[:, ref_elem.R]
    mesh = UniformMesh1D(xl, xr, num_elements, ref_elem, periodic=False)

    final_time = 0.1
    dt = 1e-5

    term = diffrax.ODETerm(rhs)
    solver = LSERK4()
    u0 = vmap(vmap(initial_condition))(mesh.x)
    solution = diffrax.diffeqsolve(
        term,
        solver,
        t0=0,
        t1=final_time,
        dt0=dt,
        y0=u0,
        args=(mesh,),
        max_steps=None,
        progress_meter=diffrax.TqdmProgressMeter(),
    )
    u_final = solution.ys[-1]  # (K, Np, 4)

    _u_exact = vmap(vmap(u_exact, (0, None)), (0, None))(
        mesh.x, final_time
    )  # (K, Np, 4)

    resid_2 = (u_final - _u_exact) ** 2  # (K, Np, 4)
    w = ref_elem.w  # (Np,)
    h = (xr - xl) / num_elements
    err = jnp.sqrt(jnp.einsum("n,knm->km", 0.5 * h * w, resid_2).sum())

    return err.item()


def generate_data():
    ps = [1, 2, 3, 4, 5]
    Ks = [5, 10, 20, 40, 80]

    out = {}

    for p in ps:
        out[p] = {"x": [], "y": []}
        for K in Ks:
            err = solve_and_compute_error(K, p)
            h = (xr - xl) / K
            out[p]["x"].append(h)
            out[p]["y"].append(err)

    with open("./data/eulerquasi1d_data.json", "w") as f:
        f.write(json.dumps(out, indent=4))


def plot_data():
    import matplotlib.pyplot as plt

    data = json.load(open("./data/eulerquasi1d_data.json", "r"))

    for p in data:
        x, y = data[p]["x"], data[p]["y"]
        rate = get_least_squares_rate(x[-3:], y[-3:])
        plt.scatter(x, y, label=f"p={p}, rate={rate:.2f}")
        plt.plot(x, y)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("h")
    plt.ylabel(r"$L^2 error$")
    plt.legend()
    plt.savefig("eulerquasi1d_accuracy.png", dpi=200)


if __name__ == "__main__":
    # generate_data()
    plot_data()
