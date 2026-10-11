from abc import ABC, abstractmethod
from collections import namedtuple
from functools import partial
import json
import math
import os
from typing import Callable, Literal, Optional, Sequence

import diffrax
import jax
from jax import jit, vmap
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)

from sbplite4py.dissipation import dissipation_local_lax_friedrichs
from sbplite4py.equations import EulerQuasi1D
from sbplite4py.equations.eulerquasi1d import flux_chan_et_al
from sbplite4py.flux_difference import evaluate_weak_flux_derivative_1d
from sbplite4py.mesh import UniformMesh1D
from sbplite4py.ref_elem import LegendreGaussLobatto1D
from sbplite4py.sats.weak import get_weak_form_sats_1d
from sbplite4py.utils.errors import compute_error_wrt_fine_mesh_solution_1d
from sbplite4py.utils.odeint import CflStepSizeController, LSERK4
from sbplite4py.utils.polynomials import eval_lagrange
from sbplite4py.postprocessing import plot_1d_timseries_as_heatmap
from sbplite4py.postprocessing.dg import plot_1d as plot_1d_dg_solution

import fourierfeatures1d


Array = jax.Array
InflowDensity = float
InflowPressure = float
OutflowPressure = float
Control = Callable[[Array, Array, "StaticArgs"], Array]  # (t, u, args)
StaticArgs = tuple[
    EulerQuasi1D,
    UniformMesh1D,
    InflowDensity,
    InflowPressure,
    OutflowPressure,
    Optional[Control],
]


# Nozzle geometry
xl = 0
xr = 5
area_fn = lambda x: 1 + 0.2 * jnp.cos(2 * jnp.pi * x / (xr - xl))  # noqa


final_time = 100

# Equation being solved
equation = EulerQuasi1D(gamma=1.4, nondimensionalize=True)

x_min = xl
x_max = xr
x_mean = 0.5 * (x_min + x_max)
x_std = (x_max - x_min) / math.sqrt(12)

t_min = 0
t_max = final_time
t_mean = 0.5 * (t_min + t_max)
t_std = (t_max - t_min) / math.sqrt(12)

pb_min = 0.96
pb_max = 1.0
pb_mean = 0.5 * (pb_min + pb_max)
pb_std = (pb_max - pb_min) / math.sqrt(12)

# Statistics of outputs from dataset
rho_mean = 0.9858370225242842
rho_std = 0.01933918467407858
v1_mean = 0.14304131024116465
v1_std = 0.12828394688375902
p_mean = 0.9803348795367415
p_std = 0.02664763128416731


class Figure(ABC):
    def __init__(self, name: str, caption: str):
        self._name = name
        self._caption = caption

    @property
    def name(self) -> str:
        return self._name

    @property
    def caption(self) -> str:
        return self._caption

    @abstractmethod
    def generate_data(
        self, checkpoint: bool = False, filepath: Optional[str] = None
    ) -> None:
        pass

    @abstractmethod
    def load_data(self, filepath: Optional[str] = None) -> None:
        pass

    @abstractmethod
    def save_data(self, filepath: Optional[str] = None) -> None:
        pass

    @abstractmethod
    def plot(self) -> None:
        pass


class IdentityControl:
    def __init__(self, gain, surrogate):
        self.surrogate = surrogate
        self.gain = gain
        self.cfl = 0.1

    def __call__(self, t, u, args: StaticArgs):
        mesh = args[1]

        # Get gain using timestep via CFL condition
        max_wave_speed = jnp.max(equation.max_wave_speed(u))
        dt = self.cfl * mesh.h / max_wave_speed
        gain = 3 / dt

        r = self.surrogate(mesh.x, t) - u
        return gain * r


def dudw(w):
    if len(w.shape) == 1:
        return jax.jacobian(equation.entropy_to_conserved)(w)

    batch_shape = w.shape[:-1]
    w = w.reshape(-1, w.shape[-1])
    uw = jax.vmap(jax.jacobian(equation.entropy_to_conserved))(w)
    uw = uw.reshape(*batch_shape, uw.shape[-2], uw.shape[-1])

    return uw


class WeaklyStableControl:
    def __init__(self, gain, surrogate):
        self.surrogate = surrogate
        self.gain = gain
        self.cfl = 0.1

    def __call__(self, t, u, args: StaticArgs):
        mesh = args[1]

        # Get gain using timestep via CFL condition
        max_wave_speed = jnp.max(equation.max_wave_speed(u))
        dt = self.cfl * mesh.h / max_wave_speed
        gain = 3 / dt

        u_nn = self.surrogate(mesh.x, t)

        w = equation.conserved_to_entropy(u)
        w_nn = equation.conserved_to_entropy(u_nn)
        uw = dudw(w_nn)

        sqrt_P = np.sqrt(np.expand_dims(mesh.J * mesh.ref_elem.P, axis=-1))

        r = jnp.einsum("...ij,...j->...i", uw, sqrt_P * (w_nn - w)) / sqrt_P

        return gain * r


class ConservativeControl:
    def __init__(self, gain, surrogate):
        self.gain = gain
        self.surrogate = surrogate
        self.cfl = 0.1

    def __call__(self, t, u, args: StaticArgs):
        mesh = args[1]

        # Get gain using timestep via CFL condition
        max_wave_speed = jnp.max(equation.max_wave_speed(u))
        dt = self.cfl * mesh.h / max_wave_speed
        gain = 3 / dt

        r = self.surrogate(mesh.x, t) - u

        # Project r onto orthogonal (wrt P-norm) complement of span{1}
        P = np.expand_dims(mesh.J * mesh.ref_elem.P, axis=-1)
        r_proj = r - jnp.sum(P * r, axis=-2, keepdims=True) / P.sum(-2, keepdims=True)

        return gain * r_proj


class ConservativeAndWeaklyStableControl:
    def __init__(self, gain, surrogate):
        self.gain = gain
        self.surrogate = surrogate
        self.cfl = 0.1

    def __call__(self, t, u, args: StaticArgs):
        mesh = args[1]

        # Get gain using timestep via CFL condition
        max_wave_speed = jnp.max(equation.max_wave_speed(u))
        dt = self.cfl * mesh.h / max_wave_speed
        gain = 3 / dt

        P = np.expand_dims(mesh.J * mesh.ref_elem.P, axis=-1)

        u_nn = self.surrogate(mesh.x, t)

        w = equation.conserved_to_entropy(u)
        w_nn = equation.conserved_to_entropy(u_nn)
        uw = dudw(w_nn)

        r = w_nn - w
        r_proj = r - jnp.sum(P * r, axis=-2, keepdims=True) / P.sum(-2, keepdims=True)

        r = jnp.einsum("...ij,...j->...i", uw, r_proj)
        r_proj = r - jnp.sum(P * r, axis=-2, keepdims=True) / P.sum(-2, keepdims=True)

        return gain * r_proj


class ConservativeAndStableControl:
    def __init__(self, gain, surrogate):
        self.gain = gain
        self.surrogate = surrogate
        self.cfl = 0.1

    def _get_orthonormal_basis_for_span_1u(self, u):
        a = jnp.stack((jnp.ones_like(u), u), axis=-1)  # (..., K, Np, 2)
        q, _ = jnp.linalg.qr(a, mode="reduced")
        q1, q2 = jnp.unstack(q, axis=-1)
        return q1, q2

    def __call__(self, t, u, args: StaticArgs):
        mesh = args[1]

        # Get gain using timestep via CFL condition
        max_wave_speed = jnp.max(equation.max_wave_speed(u))
        dt = self.cfl * mesh.h / max_wave_speed
        gain = 3 / dt

        P = np.expand_dims(mesh.J * mesh.ref_elem.P, axis=-1)  # (K, Np, 1)

        # Get orthonormal basis q1, q2 for span{1, u1}
        q1_1, q2_1 = self._get_orthonormal_basis_for_span_1u(u[..., 0])  # (..., K, Np)
        q1_2, q2_2 = self._get_orthonormal_basis_for_span_1u(u[..., 1])  # (..., K, Np)
        q1_3, q2_3 = self._get_orthonormal_basis_for_span_1u(u[..., 2])  # (..., K, Np)
        q1_4, q2_4 = q1_1, q2_1
        # q1_4, q2_4 = jnp.zeros_like(q1_1), jnp.zeros_like(q2_1)
        q1 = jnp.stack((q1_1, q1_2, q1_3, q1_4), axis=-1)  # (..., K, Np, 4)
        q2 = jnp.stack((q2_1, q2_2, q2_3, q2_4), axis=-1)  # (..., K, Np, 4)

        # Orthonormalize q1, q2 wrt P-norm
        q1_norm = jnp.sqrt((P * q1**2).sum(-2, keepdims=True))
        q1 = q1 / q1_norm
        q2 = q2 - jnp.sum(q1 * P * q2, axis=-2, keepdims=True) * q1
        q2_norm = jnp.sqrt((P * q2**2).sum(-2, keepdims=True))
        q2 = q2 / q2_norm

        # Project r onto orthogonal (wrt P-norm) complement of span{1, u}
        r = self.surrogate(mesh.x, t) - u
        r_proj = (
            r
            - jnp.sum(q1 * P * r, axis=-2, keepdims=True) * q1
            - jnp.sum(q2 * P * r, axis=-2, keepdims=True) * q2
        )

        r_proj = r_proj.at[..., -1].set(0.0)

        return gain * r_proj


def impose_inflow_boundary_condition(
    uf, ufp, rho_inflow, p_inflow, equation: EulerQuasi1D
):
    _, v1_inflow, _, a_inflow = equation.conserved_to_primitive(uf[0, 0])
    u_inflow = equation.primitive_to_conserved(
        rho_inflow, v1_inflow, p_inflow, a_inflow
    )
    ufp = ufp.at[0, 0].set(u_inflow)
    return ufp


def impose_outflow_boundary_condition(uf, ufp, p_outflow, equation: EulerQuasi1D):
    rho_outflow, v1_outflow, _, a_outflow = equation.conserved_to_primitive(uf[-1, -1])
    u_outflow = equation.primitive_to_conserved(
        rho_outflow, v1_outflow, p_outflow, a_outflow
    )
    ufp = ufp.at[-1, -1].set(u_outflow)
    return ufp


def compute_volume_terms(u, equation: EulerQuasi1D, mesh: UniformMesh1D):
    fx = evaluate_weak_flux_derivative_1d(u, mesh.ref_elem, equation)
    return -fx


def compute_surface_terms(uf, ufp, equation: EulerQuasi1D, mesh: UniformMesh1D):
    interface_flux = flux_chan_et_al(uf, ufp, equation)
    sats = get_weak_form_sats_1d(interface_flux, mesh.ref_elem, equation)
    diss = dissipation_local_lax_friedrichs(uf, ufp, equation)
    return sats - diss


def multiply_by_inverse_mass_matrix(du, mesh: UniformMesh1D):
    P = np.expand_dims(mesh.J * mesh.ref_elem.P, axis=-1)  # broadcast over PDE
    return du / P


def get_internal_face_state_batched(u, mesh: UniformMesh1D):
    batch_shape = u.shape[:-3]
    u = u.reshape(-1, mesh.num_elements, mesh.num_nodes, -1)
    uf = vmap(mesh.get_internal_face_state)(u)
    uf = uf.reshape(*batch_shape, mesh.num_elements, mesh.ref_elem.num_faces, -1)
    return uf


def get_external_face_state_batched(uf, mesh: UniformMesh1D):
    batch_shape = uf.shape[:-3]
    uf = uf.reshape(-1, mesh.num_elements, mesh.ref_elem.num_faces, -1)
    ufp = vmap(mesh.get_external_face_state)(uf)
    ufp = ufp.reshape(*batch_shape, mesh.num_elements, mesh.ref_elem.num_faces, -1)
    return ufp


@partial(jit, static_argnums=2)
def rhs(t, u, args: StaticArgs):
    equation, mesh, rho_inflow, p_inflow, p_outflow, control = args

    du = compute_volume_terms(u, equation, mesh)

    uf = mesh.get_internal_face_state(u)
    ufp = mesh.get_external_face_state(uf)

    ufp = impose_inflow_boundary_condition(uf, ufp, rho_inflow, p_inflow, equation)
    ufp = impose_outflow_boundary_condition(uf, ufp, p_outflow, equation)

    sats = compute_surface_terms(uf, ufp, equation, mesh)
    du = du.at[:, mesh.ref_elem.R].add(sats)

    du = multiply_by_inverse_mass_matrix(du, mesh)

    if control is not None:
        du = du + control(t, u, args)

    return du


def solve(
    equation: EulerQuasi1D,
    xl: float,
    xr: float,
    area_fn: Callable[[Array], Array],
    rho_inflow: float,
    p_inflow: float,
    p_outflow: float,
    degree: int,
    num_elements: int,
    ts: Sequence[float],
    control: Optional[Control] = None,
) -> tuple[diffrax.Solution, UniformMesh1D]:
    def initial_condition(x):
        rho = jnp.full_like(x, rho_inflow)
        p = jnp.full_like(x, p_inflow)
        v1 = jnp.full_like(x, 0.0)
        a = area_fn(x)
        u = equation.primitive_to_conserved(rho, v1, p, a)
        return u

    ref_elem = LegendreGaussLobatto1D(degree)
    mesh = UniformMesh1D(xl, xr, num_elements, ref_elem, periodic=False)

    final_time = ts[-1]
    CFL = 0.1

    term = diffrax.ODETerm(rhs)
    solver = LSERK4()
    u0 = vmap(vmap(initial_condition))(mesh.x)
    solution = diffrax.diffeqsolve(
        term,
        solver,
        t0=0,
        t1=final_time,
        dt0=None,
        y0=u0,
        args=(equation, mesh, rho_inflow, p_inflow, p_outflow, control),
        max_steps=None,
        progress_meter=diffrax.TqdmProgressMeter(),
        stepsize_controller=CflStepSizeController(mesh.h, CFL, equation),
        saveat=diffrax.SaveAt(t1=False, ts=ts),
    )

    return solution, mesh


LinearLayer = namedtuple("LinearLayer", ["W", "b"])


def save_params(params):
    params_dict = [
        {"W": np.array(layer.W), "b": np.array(layer.b)} for layer in params["linear"]
    ]
    params_dict.append({"fourier.weights": np.array(params["fourier"].weights)})
    np.savez("./data/params.npz", params_dict)


def load_params():
    paramspath = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "./data/params.npz"
    )
    params_dict = np.load(paramspath, allow_pickle=True)["arr_0"]
    params = {}
    params["linear"] = [
        LinearLayer(jnp.array(layer["W"]), jnp.array(layer["b"]))
        for layer in params_dict[:-1]
    ]

    weights = params_dict[-1]["fourier.weights"]
    params["fourier"] = fourierfeatures1d.FourierFeatures1d(
        weights, in_features=weights.shape[1], out_features=weights.shape[0]
    )

    return params


@jit
def model(params, x, t, pressure_ratio):
    # stack features
    z = jnp.array([x, t, pressure_ratio])

    # fourier features
    z = fourierfeatures1d.forward(params["fourier"], z)

    for layer in params["linear"][:-1]:
        z = jax.nn.relu(layer.W @ z + layer.b)

    z = params["linear"][-1].W @ z + params["linear"][-1].b

    rho, v1, p = z

    return rho, v1, p


def batched_model(params, x, t, pb):
    x, t, pb = jnp.broadcast_arrays(x, t, pb)

    batch_shape = x.shape

    x = x.reshape(-1)
    t = t.reshape(-1)
    pb = pb.reshape(-1)

    # Normalize inputs
    x = (x - x_mean) / x_std
    t = (t - t_mean) / t_std
    pb = (pb - pb_mean) / pb_std

    rho, v1, p = vmap(model, (None, 0, 0, 0))(params, x, t, pb)

    rho = rho.reshape(batch_shape)
    v1 = v1.reshape(batch_shape)
    p = p.reshape(batch_shape)
    a = area_fn(x * x_std + x_mean).reshape(batch_shape)

    # Un-normalize outputs
    rho = rho * rho_std + rho_mean
    v1 = v1 * v1_std + v1_mean
    p = p * p_std + p_mean

    u = equation.primitive_to_conserved(rho, v1, p, a)

    return u


def get_fine_mesh_reference_solution(
    degree: int, p_outflow: float, filepath=None
) -> tuple[Array, Array, UniformMesh1D]:
    if filepath is None:
        filepath = (
            f"./data/eulerquasi1d_fine_mesh_solution_p{degree}_pb={p_outflow}.npz"
        )

    rho_inflow = 1
    p_inflow = 1
    num_elements = 1024
    ref_elem = LegendreGaussLobatto1D(degree)
    mesh = UniformMesh1D(xl, xr, num_elements, ref_elem, periodic=False)

    if os.path.exists(filepath):
        dat = np.load(filepath)
        return dat["u"], dat["t"], mesh

    final_time = 100
    t = jnp.linspace(0, final_time, 501)

    solution, _ = solve(
        equation,
        xl,
        xr,
        area_fn,
        rho_inflow,
        p_inflow,
        p_outflow,
        degree,
        num_elements,
        t,
    )
    u = solution.ys

    np.savez(filepath, u=u, t=t)

    return u, t, mesh


class FigEulerQuasi1dErrorConvergence(Figure):
    def __init__(self, degree: int, p_outflow: float):
        super().__init__(
            name=f"fig_euler_quasi1d_error_convergence_p{degree}_pb={p_outflow}",
            caption="",
        )

        u, t, mesh = get_fine_mesh_reference_solution(degree, p_outflow)
        self.u_fine = u
        self.fine_mesh = mesh

        self.p_outflow = p_outflow
        self.degree = degree
        self.ref_elem = LegendreGaussLobatto1D(degree)
        self.num_elements = [2, 4, 8, 16, 32, 64]

        self.ts = np.linspace(0, 100, 501)

    def _get_default_filepath(self, filepath: Optional[str] = None) -> str:
        if filepath is None:
            return f"./data/{self.name}_data.npz"

        if filepath.split(".")[-1] != "npz":
            filepath = f"./data/{filepath}.npz"

        return filepath

    def _solve_and_compute_errors(
        self, num_elements: int, control=None
    ) -> tuple[float, float, float]:
        solution, mesh = solve(
            equation,
            xl,
            xr,
            area_fn,
            rho_inflow=1,
            p_inflow=1,
            p_outflow=self.p_outflow,
            degree=self.degree,
            num_elements=num_elements,
            ts=self.ts,
            control=control,
        )

        u1_pred = solution.ys[-1, :, :, 0]  # a*rho
        u2_pred = solution.ys[-1, :, :, 1]  # a*rho*v1
        u3_pred = solution.ys[-1, :, :, 2]  # a*E

        u1_exact = self.u_fine[-1, :, :, 0]  # a*rho
        u2_exact = self.u_fine[-1, :, :, 1]  # a*rho*v1
        u3_exact = self.u_fine[-1, :, :, 2]  # a*E

        u1_error = compute_error_wrt_fine_mesh_solution_1d(
            u1_pred, mesh, u1_exact, self.fine_mesh
        ).item()
        u2_error = compute_error_wrt_fine_mesh_solution_1d(
            u2_pred, mesh, u2_exact, self.fine_mesh
        ).item()
        u3_error = compute_error_wrt_fine_mesh_solution_1d(
            u3_pred, mesh, u3_exact, self.fine_mesh
        ).item()

        return u1_error, u2_error, u3_error

    def generate_data(self, checkpoint=False, filepath=None):
        self.data = {
            "baseline_errors": [],
            "identity_nudging_errors": [],
            "weakly_stable_nudging_errors": [],
            "conservative_nudging_errors": [],
            "conservative_and_weakly_stable_nudging_errors": [],
            "conservative_and_stable_nudging_errors": [],
        }

        # Compute errors for baseline scheme
        for K in tqdm(self.num_elements):
            errors = self._solve_and_compute_errors(K)
            self.data["baseline_errors"].append(errors)

        # Define surrogate
        params = load_params()

        def surrogate(x, t):
            u = batched_model(params, x, t, self.p_outflow)
            return u

        # Compute nudging errors
        for K in tqdm(self.num_elements):
            gain = None

            control = IdentityControl(gain, surrogate)
            errors = self._solve_and_compute_errors(K, control)
            self.data["identity_nudging_errors"].append(errors)

            control = WeaklyStableControl(gain, surrogate)
            errors = self._solve_and_compute_errors(K, control)
            self.data["weakly_stable_nudging_errors"].append(errors)

            control = ConservativeControl(gain, surrogate)
            errors = self._solve_and_compute_errors(K, control)
            self.data["conservative_nudging_errors"].append(errors)

            control = ConservativeAndWeaklyStableControl(gain, surrogate)
            errors = self._solve_and_compute_errors(K, control)
            self.data["conservative_and_weakly_stable_nudging_errors"].append(errors)

            control = ConservativeAndStableControl(gain, surrogate)
            errors = self._solve_and_compute_errors(K, control)
            self.data["conservative_and_stable_nudging_errors"].append(errors)

    def save_data(self, filepath=None):
        filepath = self._get_default_filepath(filepath)
        np.savez(filepath, self.data)
        print(f"Saved data to {filepath}")

    def load_data(self, filepath=None):
        filepath = self._get_default_filepath(filepath)
        data = np.load(filepath, allow_pickle=True)["arr_0"].item()
        self.data = data
        print(f"Loaded data from {filepath}")

    def plot(self):
        fig, axs = plt.subplots(
            1,
            3,
            figsize=(15, 5),
            dpi=200,
        )

        xl = 0
        xr = 5
        hs = [(xr - xl) / n for n in self.num_elements]

        for i in range(3):
            axs[i].plot(
                hs,
                [e[i] for e in self.data["baseline_errors"]],
                c="C0",
                label="baseline",
            )
            axs[i].plot(
                hs,
                [e[i] for e in self.data["identity_nudging_errors"]],
                c="C1",
                label="identity",
            )
            axs[i].plot(
                hs,
                [e[i] for e in self.data["weakly_stable_nudging_errors"]],
                c="C2",
                ls="--",
                label="weakly-stable",
            )
            axs[i].plot(
                hs,
                [e[i] for e in self.data["conservative_nudging_errors"]],
                c="C3",
                label="conservative",
            )
            axs[i].plot(
                hs,
                [
                    e[i]
                    for e in self.data["conservative_and_weakly_stable_nudging_errors"]
                ],
                c="C4",
                ls="--",
                label="conservative+weakly-stable",
            )
            axs[i].plot(
                hs,
                [e[i] for e in self.data["conservative_and_stable_nudging_errors"]],
                c="C5",
                label="conservative+entropy-stable",
            )
            axs[i].set_xscale("log")
            axs[i].set_yscale("log")
            axs[i].set_xlabel("h")
            axs[i].set_ylabel(r"$L^2$ error")

            if i == 2:
                axs[i].legend(loc="lower right")

        axs[0].set_title(r"$u_1 = a\rho$")
        axs[1].set_title(r"$u_2 = a\rho u$")
        axs[2].set_title(r"$u_3 = a E$")

        # plt.show()

        plot_filepath = f"./plots/{self.name}.jpg"
        plt.savefig(plot_filepath)
        print(f"Saved plot to {plot_filepath}")


if __name__ == "__main__":
    fig = FigEulerQuasi1dErrorConvergence(degree=3, p_outflow=0.98)
    # fig.generate_data()
    # fig.save_data()
    fig.load_data()
    fig.plot()
