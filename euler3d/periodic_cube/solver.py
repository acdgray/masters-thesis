from functools import partial
from typing import Callable, Optional

import diffrax
from jax import jit, vmap
from jax import Array
import jax.numpy as jnp
import numpy as np


from sbplite4py.curvilinear.transformations import ramirez_et_al
from sbplite4py.mesh import CurvilinearMesh3D
from sbplite4py.ref_elem import LegendreGaussLobatto3D
from sbplite4py.typing import TwoPointFlux
from sbplite4py.equations import Euler3D
from sbplite4py.equations.euler3d import flux_ismail_roe, residual
from sbplite4py.utils.odeint import LSERK4, CflStepSizeController
from sbplite4py.flux_difference import (
    evaluate_flux_derivative_3d,
    evaluate_weak_flux_derivative_3d,
)
from sbplite4py.sats.strong import get_strong_form_sats_3d
from sbplite4py.sats.weak import get_weak_form_sats_3d
from sbplite4py.dissipation import dissipation_local_lax_friedrichs_entropy_variables


# (xyz,t,equation: Euler3D) -> u
SourceTerm = Callable[[Array, Array, Euler3D], Array]

# (xyz) -> u
InitialCondition = Callable[[Array], Array]

StaticArgs = tuple[Euler3D, CurvilinearMesh3D, Optional[SourceTerm], TwoPointFlux, bool]

final_time = 1
equation = Euler3D(gamma=1.4)


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


@partial(jit, static_argnums=2)
def rhs_strong_form(t, u, args: StaticArgs):
    equation, mesh, source_term, two_point_flux_fn, dissipation = args

    # volume terms
    fx = evaluate_flux_derivative_3d(
        u, mesh.mt, mesh.mtf, mesh.mj, mesh.ref_elem, equation, flux=two_point_flux_fn
    )
    du = -fx.sum(-1)

    # surface terms
    uf = mesh.get_internal_face_state(u)
    ufp = mesh.get_external_face_state(uf)
    f = equation.flux(uf)
    fstar = two_point_flux_fn(uf, ufp, equation)
    sats = get_strong_form_sats_3d(f, fstar, mesh.mtf, mesh.mj, mesh.ref_elem, equation)

    if dissipation:
        d = dissipation_local_lax_friedrichs_entropy_variables(
            uf, ufp, mesh.mtf, mesh.ref_elem, equation
        )
        Pf = jnp.expand_dims(mesh.mjf * mesh.ref_elem.P[mesh.ref_elem.R], axis=-1)
        # multiply dissipation by inverse mass matrix
        surface_terms = sats - d / Pf

    else:
        surface_terms = sats

    R = mesh.ref_elem.R
    du = du.at[:, :, :, R[0], R[1], R[2]].add(surface_terms)

    if source_term is not None:
        du = du + source_term(mesh.xyz, t, equation)

    return du


@partial(jit, static_argnums=2)
def rhs_weak_form(t, u, args: StaticArgs):
    equation, mesh, source_term, two_point_flux_fn, dissipation = args

    # volume terms
    fx = evaluate_weak_flux_derivative_3d(
        u, mesh.mt, mesh.ref_elem, equation, flux=two_point_flux_fn
    )
    du = -fx.sum(-1)

    # surface terms
    uf = mesh.get_internal_face_state(u)
    ufp = mesh.get_external_face_state(uf)
    fstar = two_point_flux_fn(uf, ufp, equation)
    sats = get_weak_form_sats_3d(fstar, mesh.mtf, mesh.ref_elem, equation)

    if dissipation:
        d = dissipation_local_lax_friedrichs_entropy_variables(
            uf, ufp, mesh.mtf, mesh.ref_elem, equation
        )
        surface_terms = sats - d
    else:
        surface_terms = sats

    R = mesh.ref_elem.R
    du = du.at[:, :, :, R[0], R[1], R[2]].add(surface_terms)

    # Multiply by inverse mass matrix
    P = jnp.expand_dims(mesh.mj * mesh.ref_elem.P, axis=-1)
    du = du / P

    if source_term is not None:
        du = du + source_term(mesh.xyz, t, equation)

    return du


@partial(jit, static_argnums=2)
def strong_form_statistics_fn(t, u, args: StaticArgs):
    equation, mesh, _, _, _ = args

    P = jnp.expand_dims(mesh.mj * mesh.ref_elem.P, axis=-1)

    w = equation.conserved_to_entropy(u)

    du = rhs_strong_form(t, u, args)

    ds = (w * P * du).sum()
    du = (P * du).sum((0, 1, 2, 3, 4, 5))

    s = equation.entropy(u)
    s = (P.squeeze() * s).sum()

    u = (P * u).sum((0, 1, 2, 3, 4, 5))

    out = {"u": u, "du": du, "s": s, "ds": ds}

    return out


@partial(jit, static_argnums=2)
def weak_form_statistics_fn(t, u, args: StaticArgs):
    equation, mesh, _, _, _ = args

    P = jnp.expand_dims(mesh.mj * mesh.ref_elem.P, axis=-1)

    w = equation.conserved_to_entropy(u)

    du = rhs_weak_form(t, u, args)

    ds = (w * P * du).sum()
    du = (P * du).sum((0, 1, 2, 3, 4, 5))

    s = equation.entropy(u)
    s = (P.squeeze() * s).sum()

    u = (P * u).sum((0, 1, 2, 3, 4, 5))

    out = {"u": u, "du": du, "s": s, "ds": ds}

    return out


def comp2phys(xyz):

    # maps from [-1, 1]^3 to [0, 3]^3
    xyz = ramirez_et_al(xyz)

    # Map from [0, 3]^3 back to [-1, 1]^3
    xyz = 2 / 3 * xyz - 1

    return xyz


def _solve(
    initial_condition: InitialCondition,
    equation: Euler3D,
    num_elements: int,
    degree: int,
    final_time: float,
    CFL: float,
    curvilinear: bool = False,
    strong_form: bool = True,
    two_point_flux_fn: TwoPointFlux = flux_ismail_roe,
    source_term: Optional[SourceTerm] = None,
    dissipation: bool = False,
) -> tuple[diffrax.Solution, CurvilinearMesh3D]:
    r"""Solves the periodic 3d compressible Euler equations in the cube :math:`[-1, 1]^3`.

    Args:
        initial_condiiton (InitialCondition): Initial condition.
        equation (Euler3D): Equation being solved.
        num_elements (int): Number of elements along each axis.
        degree (int): Degree of LGL tensor-product element.
        final_time (float): Solve until this time.
        CFL (float): CFL number in :math:`(0, 1)`.
        curvilinear (bool): If ``True`` solve in curvilinear coordinates.
        strong_form (bool): If ``True``, use a strong-form-type discretization.
        two_point_flux_fn (TwoPointFlux): Numerical flux for volume flux and surface flux.
        source_term (SourceTerm, optional): If provided, add this source term.
        dissipation (bool): If ``True``, apply entropy-variables interface dissipation.

    Returns:
        tuple[diffrax.Solution, CurvilinearMesh3D]: Solution object containing the numerical solution at the final
            time and running statistics from the time integration, as well as the mesh used in the solve.
    """
    ref_elem = LegendreGaussLobatto3D(degree, indexing="ij")
    vx = vy = vz = np.linspace(-1, 1, num_elements + 1)
    mapping = comp2phys if curvilinear else lambda xyz: xyz
    mesh = CurvilinearMesh3D(vx, vy, vz, ref_elem, mapping, indexing="ij")

    u0 = initial_condition(mesh.xyz)
    static_args = (equation, mesh, source_term, two_point_flux_fn, dissipation)

    rhs = rhs_strong_form if strong_form else rhs_weak_form

    # call once to raise errors outside of diffrax since those errors are unhelpful
    _ = rhs(0.0, u0, static_args)

    term = diffrax.ODETerm(rhs)
    solver = LSERK4()

    final_subsaveat = diffrax.SubSaveAt(t1=True)
    evolving_subsaveat = diffrax.SubSaveAt(
        ts=jnp.linspace(0, final_time, 500),
        fn=strong_form_statistics_fn if strong_form else weak_form_statistics_fn,
    )

    solution = diffrax.diffeqsolve(
        term,
        solver,
        t0=0,
        t1=final_time,
        dt0=None,
        y0=u0,
        args=static_args,
        max_steps=None,
        progress_meter=diffrax.TqdmProgressMeter(),
        stepsize_controller=CflStepSizeController(
            cfl=CFL, mesh_size=mesh.h, equation=equation
        ),
        saveat=diffrax.SaveAt(subs=[final_subsaveat, evolving_subsaveat]),
    )

    return solution, mesh


def solve(
    N: int, p: int, dissipation: bool
) -> tuple[diffrax.Solution, CurvilinearMesh3D]:
    return _solve(
        initial_condition_fn,
        equation,
        num_elements=N,
        degree=p,
        final_time=final_time,
        CFL=0.1,
        curvilinear=True,
        strong_form=False,
        source_term=source_term_fn,
        dissipation=dissipation,
    )


def compute_error(solution: diffrax.Solution, mesh: CurvilinearMesh3D) -> float:
    x, y, z = jnp.unstack(mesh.xyz, axis=-1)

    u_pred = solution.ys[0][-1]
    u_exact = manufactured_solution_fn(x, y, z, final_time)

    P = mesh.J * mesh.ref_elem.P

    return jnp.sqrt((P[..., None] * (u_exact - u_pred) ** 2).sum()).item()
