from functools import partial

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap

# Specify double precision floats before importing SBPLite4py
jax.config.update("jax_enable_x64", True)

from sbplite4py.dissipation import dissipation_local_lax_friedrichs
from sbplite4py.mesh import UniformMesh1D
from sbplite4py.ref_elem import LegendreGaussLobatto1D
from sbplite4py.flux_difference import evaluate_weak_flux_derivative_1d
from sbplite4py.equations.eulerquasi1d import EulerQuasi1D, flux_chan_et_al
from sbplite4py.utils.odeint import LSERK4


xl = 0.0
xt = 5.0
xr = 10.0
equation = EulerQuasi1D(gamma=1.4)


def area_func(x):
    return jnp.where(x <= xt, 1 + 1.5 * (1 - x / xt) ** 2, 1 + 0.5 * (1 - x / xt) ** 2)


# Boundary conditions
R = 287
Tl = 300.0  # inlet temperature (K)
Mal = 0.2395  # Mach number at inlet
p_inflow = 9.608491914104024e04  # inlet pressure (Pa)
p_outflow = 8.497381834742936e04  # back pressure (Pa)
rho_inflow = p_inflow / (R * Tl)  # inlet density (kg/m^3) via ideal gas law
cl = np.sqrt(equation.gamma * p_inflow / rho_inflow)  # speed of sound at inlet (m/s)
v1_inflow = Mal * cl  # velocity at inlet (m/s)


def initial_condition(x):
    rho = rho_inflow + 0 * x
    v1 = 0 * x
    p = p_inflow + 0 * x
    a = area_func(x)
    u = equation.primitive_to_conserved(rho, v1, p, a)
    return u


@partial(jit, static_argnums=2)
def rhs(t, u, args: tuple[UniformMesh1D]):
    (mesh,) = args
    uf_interior = mesh.get_internal_face_state(u)
    uf_exterior = mesh.get_external_face_state(uf_interior)

    # Impose inflow BC
    u_inflow = uf_interior[0, 0]
    _, v1_inflow_, _, a_inflow = equation.conserved_to_primitive(u_inflow)
    u_inflow = equation.primitive_to_conserved(
        rho_inflow, v1_inflow_, p_inflow, a_inflow
    )
    uf_exterior = uf_exterior.at[0, 0].set(u_inflow)

    # Impose outflow BC
    u_outflow = uf_interior[-1, -1]
    rho_outflow, v1_outflow, _, a_outflow = equation.conserved_to_primitive(u_outflow)
    u_outflow = equation.primitive_to_conserved(
        rho_outflow, v1_outflow, p_outflow, a_outflow
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

    du = volume_contribs + surface_contribs
    return du


def solve(K, p):
    K = int(K)
    p = int(p)

    num_elements = K
    degree = p
    num_nodes = degree + 1
    ref_elem = LegendreGaussLobatto1D(degree)
    ref_elem.LIFT = jnp.diag(1 / ref_elem.P) @ jnp.eye(num_nodes)[:, ref_elem.R]
    mesh = UniformMesh1D(xl, xr, num_elements, ref_elem, periodic=False)

    final_time = 5.0
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

    return u_final, mesh


if __name__ == "__main__":
    import cd_nozzle
    import matplotlib.pyplot as plt

    u_final, mesh = solve(64, 3)

    rho, v1, p, _ = equation.conserved_to_primitive(u_final)

    xs = np.linspace(xl, xr, 100)

    rho_ex, u_ex, p_ex = jax.vmap(
        lambda x: cd_nozzle.solve(
            x,
            area_func(x),
            rhol=rho_inflow,
            ul=v1_inflow,
            pl=p_inflow,
            pr=p_outflow,
            xt=xt,
            Al=area_func(xl),
            At=area_func(xt),
            Ar=area_func(xr),
        )
    )(xs)

    fig, axs = plt.subplots(1, 3, figsize=(10, 3), dpi=200)

    axs[0].plot(xs, rho_ex, c="black", label="exact")
    axs[0].scatter(mesh.x, rho, c="red", label="SBP")
    axs[0].set_xlabel(r"$x \quad \left[ m \right]$")
    axs[0].set_ylabel(r"$\rho \quad \left[ kg/m^3 \right]$")
    axs[0].legend()

    axs[1].plot(xs, u_ex, c="black", label="exact")
    axs[1].scatter(mesh.x, v1, c="red", label="SBP")
    axs[1].set_xlabel(r"$x \quad \left[ m \right]$")
    axs[1].set_ylabel(r"$u \quad \left[ m/s \right]$")
    axs[1].legend()

    axs[2].plot(xs, p_ex, c="black", label="exact")
    axs[2].scatter(mesh.x, p, c="red", label="SBP")
    axs[2].set_xlabel(r"$x \quad \left[ m \right]$")
    axs[2].set_ylabel(r"$p \quad \left[ Pa \right]$")
    axs[2].legend()

    plt.tight_layout()
    plt.savefig("eulerquasi1d_supersonic.png")
