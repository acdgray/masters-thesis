from functools import partial
from typing import Callable, Optional

import diffrax
import jax
from jax import jit
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

from sbplite4py.curvilinear.transformations import hicken_et_al
from sbplite4py.mesh import CurvilinearMesh2D
from sbplite4py.ref_elem import LegendreGaussLobatto2D
from sbplite4py.flux_difference import evaluate_flux_derivative_2d
from sbplite4py.sats import get_strong_form_sats_2d
from sbplite4py.dissipation import dissipation_local_lax_friedrichs_entropy_variables
from sbplite4py.equations.euler2d import Euler2D, flux_ismail_roe
from sbplite4py.utils.odeint import CflStepSizeController, LSERK4


# control(t, u, mesh)
Control = Callable[[jax.Array, jax.Array, CurvilinearMesh2D], jax.Array]

StaticArgs = tuple[CurvilinearMesh2D, Euler2D, bool, Control]

equation = Euler2D(gamma=1.4)

x0 = 5.0  # x-coordinate of the center of the vortex at t=0
y0 = 0.0  # y-coordinate of the center of the vortex at t=0
eps = 1.0  # vortex strength
M = 0.5  # Mach number

# physical domain
xmin = 0.0
xmax = 20.0
ymin = -5.0
ymax = 5.0

xminymin = jnp.array([xmin, ymin])
xmaxymax = jnp.array([xmax, ymax])


def curvilinear_transformation(rs):
    """Global curvilinear transformation defined on [xmin, xmax] x [ymin, ymax]"""

    # Map from [xmin, xmax] x [ymin, ymax] to [0, 1] x [0, 1]
    rs = (rs - xminymin) / (xmaxymax - xminymin)

    # Apply nonlinear transformation on [0, 1] x [0, 1]
    xy = hicken_et_al(rs)

    # Scale from [0, 1] x [0, 1] to [xmin, xmax] x [ymin, ymax]
    xy = xy * (xmaxymax - xminymin) + xminymin

    return xy


def exact_solution(x, y, t):
    f = 1 - (((x - x0) - t) ** 2 + y**2)

    rho = (1 - eps**2 * (equation.gamma - 1) * M**2 / jnp.pi**2 / 8 * jnp.exp(f)) ** (
        1 / (equation.gamma - 1)
    )
    v1 = 1 - 0.5 * eps * y / jnp.pi * jnp.exp(0.5 * f)
    v2 = 0.5 * eps * ((x - x0) - t) / jnp.pi * jnp.exp(0.5 * f)
    p = rho**equation.gamma / (equation.gamma * M**2)

    u = equation.primitive_to_conserved(rho, v1, v2, p)
    return u


@partial(jit, static_argnums=2)
def rhs(t, u, args: StaticArgs):
    mesh, equation, dissipation, control = args

    f = evaluate_flux_derivative_2d(
        u, mesh.mt, mesh.mtf, mesh.mj, mesh.ref_elem, equation
    )
    du = -f.sum(-1)

    uf = mesh.get_internal_face_state(u)
    ufp = mesh.get_external_face_state(uf)
    flux = equation.flux(uf)
    numerical_flux = flux_ismail_roe(uf, ufp, equation)
    SATs = get_strong_form_sats_2d(
        flux, numerical_flux, mesh.mtf, mesh.mj, mesh.ref_elem, equation
    )

    if dissipation:
        d = dissipation_local_lax_friedrichs_entropy_variables(
            uf, ufp, mesh.mtf, mesh.ref_elem, equation
        )
        Pf = jnp.expand_dims(mesh.mjf * mesh.ref_elem.P[mesh.ref_elem.R], axis=-1)

        # multiply by inverse mass matrix
        d = d / Pf

        surface_terms = SATs - d
    else:
        surface_terms = SATs

    du = du.at[:, :, mesh.ref_elem.R[0], mesh.ref_elem.R[1]].add(surface_terms)

    if control is not None:
        du = du + control(t, u, mesh)

    return du


@partial(jit, static_argnums=2)
def statistics(t, u, args: StaticArgs):
    mesh, equation, _, control = args

    if control is not None:
        ctrl = control(t, u, mesh)

    w = equation.conserved_to_entropy(u)
    du = rhs(t, u, args)
    ds = (w * du).sum(-1)

    # norm matrices defined on physical space
    P = mesh.mj * mesh.ref_elem.P

    # Integral of entropy over entire spatial domain
    total_s = (P * equation.entropy(u)).sum()

    # Integral of ds/dt over entire spatial domain
    total_ds = (P * ds).sum()

    # Integral of du/dt over entire physical domain
    total_du = (du * P[..., None]).sum((0, 1, 2, 3))

    # Integral of u over entire physical domain
    total_u = (u * P[..., None]).sum((0, 1, 2, 3))

    out = {
        "total_s": total_s,
        "total_ds": total_ds,
        "total_du": total_du,
        "total_u": total_u,
        "u": u,
        "du": du,
        "ds": ds,
    }

    if control is not None:
        out["control"] = ctrl

    return out


def solve(
    Nx: int,
    Ny: int,
    degree: int,
    final_time: float,
    CFL: float,
    dissipation: bool = False,
    control: Optional[Control] = None,
) -> tuple[diffrax.Solution, CurvilinearMesh2D]:

    def initial_condition(x, y):
        return exact_solution(x, y, 0)

    ref_elem = LegendreGaussLobatto2D(degree)
    vx = np.linspace(xmin, xmax, Nx + 1)
    vy = np.linspace(ymin, ymax, Ny + 1)
    # periodic boundary conditions
    mesh = CurvilinearMesh2D(vx, vy, ref_elem, curvilinear_transformation)

    term = diffrax.ODETerm(rhs)
    solver = LSERK4()
    mesh.x, mesh.y = jnp.unstack(mesh.xy, axis=-1)
    u0 = initial_condition(mesh.x, mesh.y)

    mesh.h = jnp.minimum(
        jnp.min(mesh.x[:, :, 0, 1] - mesh.x[:, :, 0, 0]),
        jnp.min(mesh.y[:, :, 1, 0] - mesh.y[:, :, 0, 0]),
    )

    final_subsaveat = diffrax.SubSaveAt(t1=True)
    evolving_subsaveat = diffrax.SubSaveAt(
        ts=jnp.linspace(0, final_time, 1000), fn=statistics
    )

    # static arguments for ODE solver
    rhs_args = (mesh, equation, dissipation, control)

    # call once to raise errors outside of diffrax since those errors are unhelpful
    _ = rhs(0.0, u0, rhs_args)

    solution = diffrax.diffeqsolve(
        term,
        solver,
        t0=0,
        t1=final_time,
        dt0=None,
        y0=u0,
        args=rhs_args,
        max_steps=None,
        progress_meter=diffrax.TqdmProgressMeter(),
        stepsize_controller=CflStepSizeController(
            cfl=CFL, mesh_size=mesh.h, equation=equation
        ),
        saveat=diffrax.SaveAt(subs=[final_subsaveat, evolving_subsaveat]),
    )

    return solution, mesh


def compute_error(solution: diffrax.Solution, mesh: CurvilinearMesh2D) -> float:

    u_pred = solution.ys[0][-1]
    t_final = solution.ts[0][-1]

    x, y = jnp.unstack(mesh.xy, axis=-1)
    u_exact = exact_solution(x, y, t_final)

    w = (mesh.mj * mesh.ref_elem.P)[..., None]
    err = jnp.sqrt((w * (u_exact - u_pred) ** 2).sum())

    return err.item()
