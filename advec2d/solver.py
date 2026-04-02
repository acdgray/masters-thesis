from functools import partial

import diffrax
import jax
from jax import jit
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# Specify double precision floats before importing SBPLite4py
jax.config.update("jax_enable_x64", True)

from sbplite4py.mesh import CurvilinearMesh2D  # noqa: E402
from sbplite4py.ref_elem import LegendreGaussLobatto2D  # noqa: E402
from sbplite4py.utils.odeint import LSERK4  # noqa: E402
from sbplite4py.equations.advec2d import Advec2D  # noqa: E402
from sbplite4py.curvilinear.transformations import hicken_et_al  # noqa: E402
from sbplite4py.utils.errors import get_least_squares_rate  # noqa: E402


equation = Advec2D(1, 1)


def curvilinear_transformation(xy):
    return hicken_et_al(xy)


def initial_condition(x, y):
    return jnp.sin(2 * jnp.pi * x) * jnp.sin(2 * jnp.pi * y)


def exact_solution(x, y, t):
    return initial_condition(x - t, y - t)


@partial(jit, static_argnums=1)
def diff(u, mesh):
    ur = jnp.einsum("ij,...kj->...ki", mesh.ref_elem.Dx, u)
    us = jnp.einsum("ij,...jk->...ik", mesh.ref_elem.Dy, u)
    ux = (mesh.mt[..., 0, 0] * ur + mesh.mt[..., 1, 0] * us) / mesh.mj
    uy = (mesh.mt[..., 0, 1] * ur + mesh.mt[..., 1, 1] * us) / mesh.mj
    return ux, uy


@partial(jit, static_argnums=2)
def statistics(t, u, args: tuple[CurvilinearMesh2D, float]):
    mesh, _ = args

    P = mesh.J * mesh.ref_elem.P

    energy = (P * 0.5 * u**2).sum()
    mass = (P * u).sum()

    du = rhs(t, u, args)
    energy_rate = (u * P * du).sum()
    mass_rate = (P * du).sum()

    return {
        "energy": energy,
        "energy_rate": energy_rate,
        "mass": mass,
        "mass_rate": mass_rate,
    }


@partial(jit, static_argnums=2)
def rhs(t, u, args: tuple[CurvilinearMesh2D, float]):
    mesh, alpha = args
    ux, uy = diff(u, mesh)
    du = -equation.ax * ux - equation.ay * uy

    # Compute SATs
    uf = mesh.get_internal_face_state(u)
    ufp = mesh.get_external_face_state(uf)

    # Compute physical-domain outward-facing normals
    N = jnp.einsum(
        "fj,...fkji->...fki",
        mesh.ref_elem.normals,  # (4, 2)
        mesh.mtf,  # (10, 10, 4, 5, 2, 2)
    )  # (10, 10, 4, 5, 2)

    # outward-facing normals dotted with advection velocity
    An = equation.ax * N[..., 0] + equation.ay * N[..., 1]  # (10, 10, 4, 5)

    sats = 0.5 * mesh.ref_elem.B * (An - alpha * jnp.abs(An)) * (uf - ufp)

    # Scale SATs by inverse norm matrix
    sats = sats / mesh.mjf / mesh.ref_elem.P[mesh.ref_elem.R]

    # Add SAT contribution to residual
    du = du.at[:, :, mesh.ref_elem.R[0], mesh.ref_elem.R[1]].add(sats)

    return du


def solve(final_time, degree, num_elements, alpha=1):
    ref_elem = LegendreGaussLobatto2D(degree=degree)

    # Create 10-by-10 mesh for the periodic domain [0,1]x[0,1].
    vx = vy = np.linspace(0, 1, num_elements + 1)
    mesh = CurvilinearMesh2D(vx, vy, ref_elem, curvilinear_transformation)

    # Guess a small timestep
    dt = 1e-5

    # Make timestep evenly divide time interval
    num_time_steps = jnp.ceil(final_time / dt)
    dt = final_time / num_time_steps

    # Solve ODE using diffrax
    term = diffrax.ODETerm(rhs)
    solver = LSERK4()
    x, y = jnp.unstack(mesh.xy, axis=-1)
    u0 = initial_condition(x, y)

    final_subsaveat = diffrax.SubSaveAt(t1=True)
    evolving_subsaveat = diffrax.SubSaveAt(
        ts=jnp.linspace(0, final_time, 1000), fn=statistics
    )
    saveat = diffrax.SaveAt(subs=[final_subsaveat, evolving_subsaveat])

    solution = diffrax.diffeqsolve(
        term,
        solver,
        t0=0,
        t1=final_time,
        dt0=dt,
        y0=u0,
        args=(mesh, alpha),
        max_steps=int(num_time_steps) + 1,
        saveat=saveat,
        progress_meter=diffrax.TqdmProgressMeter(),
    )

    return solution, mesh


def compute_error(
    solution: diffrax.Solution, mesh: CurvilinearMesh2D, final_time: float
) -> float:
    x, y = jnp.unstack(mesh.xy, axis=-1)
    P = mesh.J * mesh.ref_elem.P

    u_pred = solution.ys[0][-1]
    u_exact = exact_solution(x, y, final_time)

    error = jnp.sqrt((P * (u_pred - u_exact) ** 2).sum()).item()

    return error
