from dataclasses import dataclass
import os
from time import perf_counter
from typing import Callable, Optional, Sequence

import jax

jax.config.update("jax_enable_x64", True)
import diffrax
from jax import Array, vmap
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sbplite4py.curvilinear.transformations import ramirez_et_al
from sbplite4py.dissipation import dissipation_local_lax_friedrichs_entropy_variables
from sbplite4py.equations import Euler3D
from sbplite4py.equations.euler3d import flux_ismail_roe, residual
from sbplite4py.flux_difference import evaluate_weak_flux_derivative_3d
from sbplite4py.mesh import CurvilinearMesh3D
from sbplite4py.ref_elem import LegendreGaussLobatto3D
from sbplite4py.sats.weak import get_weak_form_sats_3d
from sbplite4py.utils.errors import get_least_squares_rate
from sbplite4py.utils.odeint import CflStepSizeController, LSERK4
from sbplite4py.utils.figures import Figure


SourceTerm = Callable[[Array, Array, Euler3D], Array]
StaticArgs = tuple[
    Euler3D, LegendreGaussLobatto3D, CurvilinearMesh3D, Optional[SourceTerm], bool
]


def rhs_fn(t: Array, u: Array, args: StaticArgs) -> Array:
    equation, ref_elem, mesh, source_term_fn, entropy_stable = args

    fx = evaluate_weak_flux_derivative_3d(
        u, mesh.mt, ref_elem, equation, flux=flux_ismail_roe
    )
    du = -fx.sum(-1)

    uf = mesh.get_internal_face_state(u)
    ufp = mesh.get_external_face_state(uf)
    fstar = flux_ismail_roe(uf, ufp, equation)
    sats = get_weak_form_sats_3d(fstar, mesh.mtf, ref_elem, equation)
    if entropy_stable:
        diss = dissipation_local_lax_friedrichs_entropy_variables(
            uf, ufp, mesh.mtf, ref_elem, equation
        )
        du = du.at[:, :, :, ref_elem.R[0], ref_elem.R[1], ref_elem.R[2]].add(
            sats - diss
        )
    else:
        du = du.at[:, :, :, ref_elem.R[0], ref_elem.R[1], ref_elem.R[2]].add(sats)

    # Multiply by inverse mass matrix
    P = np.expand_dims(ref_elem.P, axis=-1)
    J = jnp.expand_dims(mesh.mj, axis=-1)
    du = du / (J * P)

    if source_term_fn is not None:
        source_term = source_term_fn(mesh.xyz, t, equation)
        du = du + source_term

    return du


def statistics_fn(t: Array, u: Array, args: StaticArgs) -> dict[str, Array]:
    equation, ref_elem, mesh, _, _ = args

    s = equation.entropy(u)
    w = equation.conserved_to_entropy(u)
    du = rhs_fn(t, u, args)

    P = ref_elem.P
    J = mesh.mj
    s_int = (J * P * s).sum()

    P = jnp.expand_dims(ref_elem.P, axis=-1)
    J = jnp.expand_dims(mesh.mj, axis=-1)
    st_int = (w * J * P * du).sum()
    u_int = (J * P * u).sum([i for i in range(len(u.shape) - 1)])
    ut_int = (J * P * du).sum([i for i in range(len(u.shape) - 1)])

    return {"s_int": s_int, "st_int": st_int, "u_int": u_int, "ut_int": ut_int}


def manufactured_solution_fn(x: Array, y: Array, z: Array, t: Array) -> Array:
    rho = 2 + 0.1 * jnp.sin(jnp.pi * (x + y + z - 2 * t))
    v1 = v2 = v3 = 1 + 0 * x * y * z * t
    E = rho**2
    u = jnp.stack((rho, rho * v1, rho * v2, rho * v3, E), axis=-1)
    return u


def source_term_fn(xyz: Array, t: Array, equation: Euler3D) -> Array:
    x, y, z = jnp.unstack(xyz, axis=-1)
    x, y, z, t = jnp.broadcast_arrays(x, y, z, t)

    shape = x.shape

    x = x.reshape(-1)
    y = y.reshape(-1)
    z = z.reshape(-1)
    t = t.reshape(-1)

    gamma = equation.gamma
    nondimensionalize = equation.nondimensionalize

    source_term = vmap(residual(manufactured_solution_fn, gamma, nondimensionalize))(
        x, y, z, t
    )

    return source_term.reshape((*shape, source_term.shape[-1]))


def initial_condition_fn(xyz: Array):
    return manufactured_solution_fn(*jnp.unstack(xyz, axis=-1), jnp.array(0.0))


def solve(
    degree: int,
    final_time: float,
    num_elements_along_each_axis: int,
    entropy_stable: bool,
) -> tuple[diffrax.Solution, CurvilinearMesh3D]:

    equation = Euler3D(gamma=1.4)

    ref_elem = LegendreGaussLobatto3D(degree=degree, indexing="ij")
    vx = vy = vz = np.linspace(-1, 1, num_elements_along_each_axis + 1)
    mapping = lambda xyz: 2 / 3 * ramirez_et_al(xyz) - 1
    mesh = CurvilinearMesh3D(
        vx,
        vy,
        vz,
        ref_elem,
        mapping,
        periodic=(True, True, True),
        indexing="ij",
        metric_terms="thomas-lombard",
    )

    u0 = initial_condition_fn(mesh.xyz)

    _ = rhs_fn(0.0, u0, (equation, ref_elem, mesh, source_term_fn, entropy_stable))
    _ = statistics_fn(
        0.0, u0, (equation, ref_elem, mesh, source_term_fn, entropy_stable)
    )

    term = diffrax.ODETerm(rhs_fn)
    solver = LSERK4()

    CFL = 0.1
    stepsize_controller = CflStepSizeController(
        cfl=CFL, mesh_size=mesh.h, equation=equation
    )

    final_subsaveat = diffrax.SubSaveAt(t1=True)
    evolving_subsaveat = diffrax.SubSaveAt(
        ts=jnp.linspace(0, final_time, 500), fn=statistics_fn
    )
    saveat = diffrax.SaveAt(subs=[final_subsaveat, evolving_subsaveat])

    solution = diffrax.diffeqsolve(
        term,
        solver,
        t0=0,
        t1=final_time,
        dt0=None,
        y0=u0,
        args=(equation, ref_elem, mesh, source_term_fn, entropy_stable),
        max_steps=None,
        progress_meter=diffrax.TqdmProgressMeter(),
        stepsize_controller=stepsize_controller,
        saveat=saveat,
    )

    return solution, mesh


def compute_error(
    solution: diffrax.Solution, mesh: CurvilinearMesh3D, final_time: float
) -> float:
    J = np.expand_dims(mesh.mj, axis=-1)
    P = np.expand_dims(mesh.ref_elem.P, axis=-1)

    u_pred = solution.ys[0][-1]

    u_exact = manufactured_solution_fn(
        *jnp.unstack(mesh.xyz, axis=-1), jnp.array(final_time)
    )

    error = jnp.sqrt((J * P * (u_pred - u_exact) ** 2).sum())

    return error.item()


def solve_and_compute_error(
    degree: int,
    final_time: float,
    num_elements_along_each_axis: int,
    entropy_stable: bool,
) -> float:
    solution, mesh = solve(
        degree, final_time, num_elements_along_each_axis, entropy_stable
    )
    error = compute_error(solution, mesh, final_time)
    return error
