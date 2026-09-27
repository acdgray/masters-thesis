from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
import json
from typing import Callable, Optional, Sequence

import diffrax
import jax
import jax.numpy as jnp
from jax import vmap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, ticker
import scipy.special
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)  # use double precision

from sbplite4py.equations import Advec1D
from sbplite4py.mesh import UniformMesh1D
from sbplite4py.ref_elem import LegendreGaussLobatto1D
from sbplite4py.sats.advec1d import symmetric_sats_1d, upwind_sats_1d
from sbplite4py.utils.errors import get_least_squares_rate
from sbplite4py.utils.odeint import LSERK4
from sbplite4py.utils.polynomials import eval_lagrange, eval_legendre


Array = jax.Array
Control = Callable[[Array, Array, "StaticArgs"], Array]  # (t, u, args)
StaticArgs = tuple[Advec1D, UniformMesh1D, Optional[Control]]


# Problem definition
xl = -1
xr = 1
advection_velocity = 1
equation = Advec1D(advection_velocity)
final_time = 1
initial_condition = lambda x: jnp.exp(-100 * (x + 0.5) ** 2)
exact_solution = lambda x, t: initial_condition(x - advection_velocity * t)

# Manually computed these elsewhere

# \sqrt{\int_0^1 \int_{-1}^1 u(x,t)^2 dx dt}
# also approximately equals \sqrt{\int_{-1}^1 u(x,1)^2 dx}
exact_solution_norm = 0.3540217701378688

# surrogate error = \sqrt{\int_0^1 \int_{-1}^1 [ u(x,t) - z(x,t) ]^2 dx dt}
# error statistics calculated from a random sample of 10,000,000 different z
surrogate_error_mean_div_noise = 1.037935167999395
surrogate_error_std_div_noise = 0.3518352889823121

# surrogate error at t=1 = sqrt{\int_{-1}^1 [ u(x,1) - z(x,1) ]^2 dx}
final_surrogate_error_mean_div_noise = 1.672902887078604
final_surrogate_error_std_div_noise = 0.7343895097304207


class IdentityControl:
    def __init__(self, gain, surrogate):
        self.surrogate = surrogate
        self.gain = gain

    def __call__(self, t, u, args: StaticArgs):
        _, mesh, _ = args
        r = self.surrogate(mesh.x, t) - u
        return self.gain * r


class ConservativeControl:
    def __init__(self, gain, surrogate):
        self.gain = gain
        self.surrogate = surrogate

    def __call__(self, t, u, args: StaticArgs):
        _, mesh, _ = args

        r = self.surrogate(mesh.x, t) - u

        # Project r onto orthogonal (wrt P-norm) complement of span{1}
        P = mesh.J * mesh.ref_elem.P
        r_proj = r - jnp.sum(P * r, axis=-1, keepdims=True) / P.sum(-1, keepdims=True)

        return self.gain * r_proj


class StableControl:
    def __init__(self, gain, surrogate):
        self.gain = gain
        self.surrogate = surrogate

    def __call__(self, t, u, args: StaticArgs):
        _, mesh, _ = args

        r = self.surrogate(mesh.x, t) - u

        # Project r onto orthogonal (wrt P-norm) complement of span{u}
        P = mesh.J * mesh.ref_elem.P
        r_proj = (
            r
            - jnp.sum(u * P * r, axis=-1, keepdims=True)
            / (u * P * u).sum(-1, keepdims=True)
            * u
        )

        return self.gain * r_proj


class ConservativeAndStableControl:
    def __init__(self, gain, surrogate):
        self.gain = gain
        self.surrogate = surrogate

    def _get_orthonormal_basis_for_span_1u(self, u):
        a = jnp.stack((jnp.ones_like(u), u), axis=-1)  # (..., K, Np, 2)
        q, _ = jnp.linalg.qr(a, mode="reduced")
        q1, q2 = jnp.unstack(q, axis=-1)
        return q1, q2

    def __call__(self, t, u, args: StaticArgs):
        _, mesh, _ = args

        P = mesh.J * mesh.ref_elem.P  # (K, Np)

        # Get orthonormal basis q1, q2 for span{1, u}
        q1, q2 = self._get_orthonormal_basis_for_span_1u(u)  # (..., K, Np)

        # Orthonormalize q1, q2 wrt P-norm
        q1_norm = jnp.sqrt((P * q1**2).sum(-1, keepdims=True))
        q1 = q1 / q1_norm
        q2 = q2 - jnp.sum(q1 * P * q2, axis=-1, keepdims=True) * q1
        q2_norm = jnp.sqrt((P * q2**2).sum(-1, keepdims=True))
        q2 = q2 / q2_norm

        # Project r onto orthogonal (wrt P-norm) complement of span{1, u}
        r = self.surrogate(mesh.x, t) - u
        r_proj = (
            r
            - jnp.sum(q1 * P * r, axis=-1, keepdims=True) * q1
            - jnp.sum(q2 * P * r, axis=-1, keepdims=True) * q2
        )

        return self.gain * r_proj


def get_internal_face_state_batched(u: Array, mesh: UniformMesh1D):
    batch_shape = u.shape[:-2]
    u = u.reshape(-1, mesh.num_elements, mesh.ref_elem.num_nodes)
    uf = vmap(mesh.get_internal_face_state)(u)
    uf = uf.reshape(*batch_shape, mesh.num_elements, mesh.ref_elem.num_faces)
    return uf


def get_external_face_state_batched(uf: Array, mesh: UniformMesh1D):
    batch_shape = uf.shape[:-2]
    uf = uf.reshape(-1, mesh.num_elements, mesh.ref_elem.num_faces)
    ufp = vmap(mesh.get_external_face_state)(uf)
    ufp = ufp.reshape(*batch_shape, mesh.num_elements, mesh.ref_elem.num_faces)
    return ufp


def semidiscretization(t: Array, u: Array, args: StaticArgs) -> Array:
    """Right-hand-side function of the semidiscretization"""
    equation, mesh, control = args

    ur = jnp.einsum("ij,...j->...i", mesh.ref_elem.D, u)
    ux = mesh.rx * ur
    du = -equation.a * ux

    uf = get_internal_face_state_batched(u, mesh)
    ufp = get_external_face_state_batched(uf, mesh)
    # SATs = symmetric_sats_1d(uf, ufp, mesh.J, mesh.ref_elem, equation)
    SATs = upwind_sats_1d(uf, ufp, mesh.J, mesh.ref_elem, equation)

    du = du.at[..., mesh.ref_elem.R].add(SATs)

    if control is not None:
        du = du + control(t, u, args)

    return du


def statistics(t: Array, u: Array, args: StaticArgs) -> dict[str, Array]:
    """Computes running statistics of the solution during the solve"""
    _, mesh, control = args

    du = semidiscretization(t, u, args)

    P = mesh.J * mesh.ref_elem.P

    dot = lambda u, v: (u * P * v).sum((-1, -2))
    norm = lambda u: jnp.sqrt(dot(u, u))

    out = {}
    out["mass_rate"] = dot(1, du)
    out["energy_rate"] = dot(u, du)
    out["error"] = norm(u - exact_solution(mesh.x, t))
    out["energy"] = norm(0.5 * u**2)
    out["||du/dt||"] = norm(du)
    out["||u_pred||"] = norm(u)
    out["||u_exact||"] = norm(exact_solution(mesh.x, t))

    if control is not None:
        c = control(t, u, args)
        out["||control||"] = norm(c)
        out["||-Du + SATs||"] = norm(du - c)  # du = -Du + SATs + c

        surrogate = control.surrogate(mesh.x, t)
        u_exact = exact_solution(mesh.x, t)
        u_pred = u
        out["approximate_surrogate_error"] = norm(u_pred - surrogate)
        out["exact_surrogate_error"] = norm(u_exact - surrogate)

    return out


def solve(
    xl: float,
    xr: float,
    advection_velocity: float,
    final_time: float,
    initial_condition: Callable[[Array], Array],
    num_elements: int,
    degree: int,
    control: Optional[Control] = None,
    batch_shape: Optional[tuple[int, ...]] = None,
) -> tuple[diffrax.Solution, UniformMesh1D, Array]:

    equation = Advec1D(advection_velocity)
    exact_solution = lambda x, t: initial_condition(x - advection_velocity * t)

    # Discretize space and time
    ref_elem = LegendreGaussLobatto1D(degree=degree)
    mesh = UniformMesh1D(xl, xr, num_elements, ref_elem, periodic=True)
    CFL = 0.1
    dt = CFL * mesh.h / advection_velocity

    # Set maximally stable gain
    if control is not None and control.gain is None:
        control.gain = 3 / dt

    # Solve ODE
    u0 = initial_condition(mesh.x)  # (K, Np)
    if batch_shape is not None:
        u0 = jnp.tile(u0, (*batch_shape, 1, 1))  # (..., K, Np)

    term = diffrax.ODETerm(semidiscretization)
    solver = LSERK4()
    solution = diffrax.diffeqsolve(
        term,
        solver,
        t0=0,
        t1=final_time,
        dt0=dt,
        y0=u0,
        args=(equation, mesh, control),
        max_steps=None,
        stepsize_controller=diffrax.ConstantStepSize(),
        progress_meter=diffrax.TqdmProgressMeter(),
        saveat=diffrax.SaveAt(
            subs=[
                diffrax.SubSaveAt(t1=True),
                diffrax.SubSaveAt(ts=jnp.linspace(0, final_time, 501), fn=statistics),
            ]
        ),
    )

    # Compute error at final time
    P = mesh.J * mesh.ref_elem.P
    u_pred = solution.ys[0][-1]
    u_exact = exact_solution(mesh.x, final_time)
    error = np.sqrt((P * (u_pred - u_exact) ** 2).sum((-2, -1)))

    return solution, mesh, error


def randfun(x, t, coef):
    coef = np.expand_dims(coef, axis=(-4, -3))
    x_evals = jnp.stack(
        [eval_legendre(x, i) / (i + 1) for i in range(coef.shape[-2])], axis=-1
    )
    t_evals = jnp.stack(
        [eval_legendre(t, i) / (i + 1) for i in range(coef.shape[-1])], axis=-1
    )
    return (coef * x_evals[..., None] * t_evals[..., None, :]).sum((-2, -1))


def plot_solution(solution: diffrax.Solution, mesh: UniformMesh1D) -> None:
    xs = np.linspace(xl, xr, 1000)
    plt.plot(xs, exact_solution(xs, final_time), label="exact solution", c="black")
    plt.plot([], [], label="numerical solution", c="red")
    for k in range(mesh.num_elements):
        xs = np.linspace(mesh.x[k, 0], mesh.x[k, -1], 50)
        us = eval_lagrange(xs, mesh.x[k], solution.ys[0][0, k])
        plt.plot(xs, us, c="red")
    plt.title(f"Solution at t={final_time}")
    plt.xlabel("x")
    plt.show()

    plt.title("L2-norms as a function of time")
    # plt.plot(solution.ts[1], solution.ys[1]["||u||"], label="||u||")
    plt.plot(solution.ts[1], solution.ys[1]["||du/dt||"], label="||du/dt||")
    if "||control||" in solution.ys[1]:
        plt.plot(solution.ts[1], solution.ys[1]["||control||"], label="||control||")
        plt.plot(
            solution.ts[1], solution.ys[1]["||-Du + SATs||"], label="||-Du + SATs||"
        )
        # plt.plot(solution.ts[1], solution.ys[1]["angle"], label="angle")
    plt.xlabel("t")
    plt.legend()
    plt.show()

    plt.title("Mass rate")
    plt.plot(solution.ts[1], solution.ys[1]["mass_rate"])
    plt.xlabel("t")
    plt.show()

    plt.title("Energy rate")
    plt.plot(solution.ts[1], solution.ys[1]["energy_rate"])
    plt.xlabel("t")
    plt.show()

    plt.title("Error")
    plt.plot(solution.ts[1], solution.ys[1]["error"])
    plt.xlabel("t")
    plt.show()


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


class FigAdvec1dErrorConvergence(Figure):
    def __init__(self):
        super().__init__(name="advec1d_error_convergence", caption="")

        # problem definition
        self.xl = -1
        self.xr = 1
        self.advection_velocity = 1
        self.final_time = 1
        self.initial_condition = lambda x: -jnp.sin(jnp.pi * x)
        self.exact_solution = lambda x, t: self.initial_condition(
            x - self.advection_velocity * t
        )

        # sequence of meshes for each element degree
        self.cases = [
            {"degree": 1, "num_elements": [i for i in range(9, 21)]},
            {"degree": 2, "num_elements": [i for i in range(9, 19)]},
            {"degree": 3, "num_elements": [i for i in range(9, 17)]},
            {"degree": 4, "num_elements": [i for i in range(9, 15)]},
            {"degree": 5, "num_elements": [i for i in range(9, 13)]},
        ]

        # holds the data that is plotted
        self.data = []

    def _get_default_filepath(self, filepath: Optional[str] = None) -> str:
        if filepath is None:
            return f"{self.name}_data.json"

        if filepath.split(".")[-1] != "json":
            return f"{filepath}.json"

        return filepath

    def _solve_and_compute_error(self, num_elements: int, degree: int) -> float:
        _, _, error = solve(
            self.xl,
            self.xr,
            self.advection_velocity,
            self.final_time,
            self.initial_condition,
            num_elements,
            degree,
        )
        return error

    def generate_data(
        self, checkpoint: bool = False, filepath: Optional[str] = None
    ) -> None:
        filepath = self._get_default_filepath(filepath)

        self.data = []
        for case in self.cases:
            print(f"Running error convergence for p={case['degree']}...")
            self.data.append({})
            self.data[-1]["degree"] = case["degree"]
            self.data[-1]["errors"] = []
            self.data[-1]["element_sizes"] = [
                (self.xr - self.xl) / n for n in case["num_elements"]
            ]

            for num_elements in tqdm(case["num_elements"]):
                error = self._solve_and_compute_error(
                    num_elements, degree=case["degree"]
                )
                self.data[-1]["errors"].append(error)

            self.data[-1]["rate"] = get_least_squares_rate(
                self.data[-1]["element_sizes"], self.data[-1]["errors"]
            )
            if checkpoint:
                self.save_data(filepath)

    def save_data(self, filepath: Optional[str] = None) -> None:
        filepath = self._get_default_filepath(filepath)

        with open(filepath, "w") as f:
            json.dump(self.data, f)

        print(f"Saved data to {filepath}")

    def load_data(self, filepath: Optional[str] = None) -> None:
        filepath = self._get_default_filepath(filepath)

        with open(filepath, "r") as f:
            self.data = json.load(f)

        print(f"Loaded data from {filepath}")

    def plot(self) -> None:
        for convergence_test in self.data:
            p = convergence_test["degree"]
            es = convergence_test["errors"]
            hs = convergence_test["element_sizes"]
            rate = convergence_test["rate"]
            plt.plot(hs, es)
            plt.scatter(hs, es, label=f"p={p}, rate={rate:.2f}")
        plt.xlabel("h")
        plt.ylabel(r"$L^2$ error")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend()
        plt.savefig(self.name)
        try:
            plt.show()
        except:  # noqa
            pass


class FigAdvec1dConservationAndStability(Figure):
    def __init__(self):
        super().__init__(name="advec1d_conservation_and_stability", caption="")

        # problem definition
        self.xl = -1
        self.xr = 1
        self.advection_velocity = 1
        self.final_time = 1
        self.initial_condition = lambda x: -jnp.sin(jnp.pi * x)
        self.exact_solution = lambda x, t: self.initial_condition(
            x - self.advection_velocity * t
        )

    def save_data(self, filepath: Optional[str] = None) -> None:
        raise NotImplementedError()

    def load_data(self, filepath: Optional[str] = None) -> None:
        raise NotImplementedError()

    def generate_data(self, checkpoint=False, filepath=None):
        solution, mesh, _ = solve(
            self.xl,
            self.xr,
            self.advection_velocity,
            self.final_time,
            self.initial_condition,
            num_elements=10,
            degree=4,
        )
        self.solution = solution
        self.mesh = mesh

    def plot(self) -> None:
        fig, axs = plt.subplots(1, 2)

        axs[0].plot(self.solution.ts[1], self.solution.ys[1]["mass_rate"])
        axs[0].set_xlabel("time")

        axs[1].plot(self.solution.ts[1], self.solution.ys[1]["energy_rate"])
        axs[1].set_xlabel("time")

        plt.savefig(self.name)

        try:
            plt.show()
        except:  # noqa
            pass


class LogTwoSlopeNorm(colors.FuncNorm):
    def __init__(self, vcenter, vmin=None, vmax=None):

        def forward(x):
            a = np.where(
                x < vcenter,
                0.5 / np.log10(vcenter / vmin),
                -0.5 / np.log10(vcenter / vmax),
            )
            b = np.where(x < vcenter, -a * np.log10(vmin), 1 - a * np.log10(vmax))
            y = a * np.log10(x) + b
            return y

        def inverse(y):
            a = np.where(
                y < 0.5, 0.5 / np.log10(vcenter / vmin), -0.5 / np.log10(vcenter / vmax)
            )
            b = np.where(y < 0.5, -a * np.log10(vmin), 1 - a * np.log10(vmax))
            x = 10 ** ((y - b) / a)
            return x

        super().__init__((forward, inverse), vmin=vmin, vmax=vmax)


class FigAdvec1dNudgingVsBaselineScheme(Figure):
    def __init__(self):
        super().__init__(name="advec1d_nuging_vs_baseline_scheme", caption="")

        self.meshes = [
            {"degree": 2, "num_elements": 15},
            {"degree": 3, "num_elements": 10},
            {"degree": 6, "num_elements": 4},
            {"degree": 10, "num_elements": 2},
        ]

        self.controls = [
            IdentityControl,
            ConservativeControl,
            ConservativeAndStableControl,
        ]
        self.control_names = ["Identity", "Conservative", "Conservative and stable"]

        self.num_samples = 10
        self.noises = 10 ** np.linspace(-4, 0, 50)
        self.gains = 10 ** np.linspace(-1, 2.8, 50)

        self.data = []

    def _get_default_filepath(self, filepath: Optional[str] = None) -> str:
        if filepath is None:
            return f"data/{self.name}_data.npz"

        if filepath.split(".")[-1] != "npz":
            filepath = f"{filepath}.npz"

        return filepath

    def _solve_and_compute_error(
        self,
        degree: int,
        num_elements: int,
        control: Optional[Control] = None,
        num_samples: Optional[int] = None,
    ) -> float:
        _, _, error = solve(
            xl,
            xr,
            advection_velocity,
            final_time,
            initial_condition,
            num_elements,
            degree,
            control,
            num_samples,
        )
        return error

    def save_data(self, filepath=None):
        filepath = self._get_default_filepath(filepath)
        np.savez(filepath, self.data)
        print(f"Saved data to {filepath}")

    def load_data(self, filepath=None):
        filepath = self._get_default_filepath(filepath)
        data = np.load(filepath, allow_pickle=True)["arr_0"]
        self.data = [
            [data[i][j] for j in range(len(self.controls))]
            for i in range(len(self.meshes))
        ]
        print(f"Loaded data from {filepath}")

    def generate_data(self):
        self.data = [
            [{} for j in range(len(self.controls))] for i in range(len(self.meshes))
        ]

        # Compute baseline errors
        for i, mesh in enumerate(self.meshes):
            num_elements = mesh["num_elements"]
            degree = mesh["degree"]
            baseline_error = self._solve_and_compute_error(degree, num_elements)
            for j in range(len(self.controls)):
                self.data[i][j]["baseline_error"] = float(baseline_error)

        # Re-use same randomly selected coefficients for all experiments
        coef = np.random.normal(size=(self.num_samples, 10, 10), loc=0, scale=1)

        for i, mesh in enumerate(self.meshes):
            degree = mesh["degree"]
            num_elements = mesh["num_elements"]
            for j, control_cls in enumerate(self.controls):
                # broadcast noises (epsilon) over samples, elements, and nodes
                noise = self.noises[:, None, None, None]

                # broadcast gains (k) over noises (epsilon), samples, elements, and nodes
                gain = self.gains[:, None, None, None, None]

                # Artificial surrogate
                surrogate = lambda x, t: (
                    exact_solution(x, t) + noise * randfun(x, t, coef)
                )
                control = control_cls(gain, surrogate)

                _, _, errors = solve(
                    xl,
                    xr,
                    advection_velocity,
                    final_time,
                    initial_condition,
                    num_elements=num_elements,
                    degree=degree,
                    control=control,
                    batch_shape=(len(self.gains), len(self.noises), self.num_samples),
                )

                assert errors.shape == (
                    len(self.gains),
                    len(self.noises),
                    self.num_samples,
                )
                self.data[i][j]["errors"] = errors

    def plot(self):
        fig, axs = plt.subplots(
            len(self.meshes), len(self.controls), sharex=True, sharey=True
        )

        # Get minimum and maximum relative errors across all mesh/control
        # combinations so that data can be plotted on the same scale.
        # z_min, z_max = np.inf, -np.inf
        # for i, _ in enumerate(self.meshes):
        #     for j, _ in enumerate(self.controls):
        #         baseline_error = self.data[i][j]["baseline_error"]
        #         errors = self.data[i][j]["errors"]
        #         z = errors.mean(-1) / baseline_error
        #         z_min = min(z_min, z.min())
        #         z_max = max(z_max, z.max())
        z_min = 1e-3
        z_max = 2e3

        # Add x-axis labels to bottommost row of plots
        for j, _ in enumerate(self.controls):
            axs[-1, j].set_xlabel(r"surrogate error")

        # Add y-axis labels to leftmost column of plots
        for i, mesh in enumerate(self.meshes):
            degree = mesh["degree"]
            num_elements = mesh["num_elements"]
            axs[i, 0].set_ylabel(f"(K={num_elements}, p={degree})\n" + r"gain $k$")

        # Add titles to topmost row of plots
        for j, control_name in enumerate(self.control_names):
            axs[0, j].set_title(control_name)

        # Create len(self.meshes)-by-len(self.controls) grid of contour plots
        levels = 10 ** np.linspace(np.log10(z_min), np.log10(z_max), 50)
        norm = LogTwoSlopeNorm(vcenter=1.0, vmin=z_min, vmax=z_max)

        for i, mesh in enumerate(self.meshes):
            for j, _ in enumerate(self.controls):
                ax = axs[i, j]

                expected_surrogate_error = (
                    final_surrogate_error_mean_div_noise * self.noises
                )
                relative_expected_surrogate_error = (
                    expected_surrogate_error / exact_solution_norm
                )
                x = relative_expected_surrogate_error

                y = self.gains

                x, y = np.meshgrid(x, y)

                x_min, x_max = x.min(), x.max()
                y_min, y_max = y.min(), y.max()

                ax.set_xscale("log")
                ax.set_yscale("log")

                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)

                baseline_error = self.data[i][j]["baseline_error"]
                errors = self.data[i][j]["errors"]
                z = errors.mean(-1) / baseline_error

                cs = ax.contourf(x, y, z, cmap="seismic", norm=norm, levels=levels)

        # Add colorbar to right of plots
        cbar = fig.colorbar(
            cs,
            ax=axs,
            # label=r"$\frac{\text{nudging error at }t=1}{\text{baseline scheme error at }t=1}$",
            extend="both",
        )
        cbar.locator = ticker.LogLocator(base=10.0, subs=(1.0,), numticks=10)
        cbar.update_ticks()
        cbar.ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10.0))

        # plt.tight_layout()
        plt.show()


class FigAdvec1dNudgingVsSurrogate(Figure):
    def __init__(self):
        super().__init__(name="advec1d_nuging_vs_surrogate", caption="")

        self.meshes = [
            {"degree": 2, "num_elements": 15},
            {"degree": 3, "num_elements": 10},
            {"degree": 6, "num_elements": 4},
            {"degree": 10, "num_elements": 2},
        ]

        self.controls = [
            IdentityControl,
            ConservativeControl,
            ConservativeAndStableControl,
        ]
        self.control_names = ["Identity", "Conservative", "Conservative and stable"]

        self.num_samples = 10
        self.noises = 10 ** np.linspace(-4, 0, 50)
        self.gains = 10 ** np.linspace(-1, 2.8, 50)

        self.data = []

    def _get_default_filepath(self, filepath: Optional[str] = None) -> str:
        if filepath is None:
            return f"data/{self.name}_data.npz"

        if filepath.split(".")[-1] != "npz":
            filepath = f"{filepath}.npz"

        return filepath

    def _solve_and_compute_error(
        self,
        degree: int,
        num_elements: int,
        control: Optional[Control] = None,
        num_samples: Optional[int] = None,
    ) -> float:
        _, _, error = solve(
            xl,
            xr,
            advection_velocity,
            final_time,
            initial_condition,
            num_elements,
            degree,
            control,
            num_samples,
        )
        return error

    def save_data(self, filepath=None):
        filepath = self._get_default_filepath(filepath)
        np.savez(filepath, self.data)
        print(f"Saved data to {filepath}")

    def load_data(self, filepath=None):
        filepath = self._get_default_filepath(filepath)
        data = np.load(filepath, allow_pickle=True)["arr_0"]
        self.data = [
            [data[i][j] for j in range(len(self.controls))]
            for i in range(len(self.meshes))
        ]
        print(f"Loaded data from {filepath}")

    def generate_data(self):
        self.data = [
            [{} for j in range(len(self.controls))] for i in range(len(self.meshes))
        ]

        # Compute baseline errors
        for i, mesh in enumerate(self.meshes):
            num_elements = mesh["num_elements"]
            degree = mesh["degree"]
            baseline_error = self._solve_and_compute_error(degree, num_elements)
            for j in range(len(self.controls)):
                self.data[i][j]["baseline_error"] = float(baseline_error)

        # Re-use same randomly selected coefficients for all experiments
        coef = np.random.normal(size=(self.num_samples, 10, 10), loc=0, scale=1)

        for i, mesh in enumerate(self.meshes):
            degree = mesh["degree"]
            num_elements = mesh["num_elements"]
            for j, control_cls in enumerate(self.controls):
                # broadcast noises (epsilon) over samples, elements, and nodes
                noise = self.noises[:, None, None, None]

                # broadcast gains (k) over noises (epsilon), samples, elements, and nodes
                gain = self.gains[:, None, None, None, None]

                # Artificial surrogate
                surrogate = lambda x, t: (
                    exact_solution(x, t) + noise * randfun(x, t, coef)
                )
                control = control_cls(gain, surrogate)

                _, mesh, errors = solve(
                    xl,
                    xr,
                    advection_velocity,
                    final_time,
                    initial_condition,
                    num_elements=num_elements,
                    degree=degree,
                    control=control,
                    batch_shape=(len(self.gains), len(self.noises), self.num_samples),
                )

                assert errors.shape == (
                    len(self.gains),
                    len(self.noises),
                    self.num_samples,
                )
                self.data[i][j]["errors"] = errors

                # Compute surrogate errors
                P = mesh.J * mesh.ref_elem.P
                u_pred = exact_solution(mesh.x, final_time) + self.noises[
                    :, None, None, None
                ] * randfun(mesh.x, final_time, coef)
                u_exact = exact_solution(mesh.x, final_time)
                surrogate_errors = np.sqrt((P * (u_pred - u_exact) ** 2).sum((-2, -1)))
                self.data[i][j]["surrogate_errors"] = surrogate_errors

    def plot(self):
        fig, axs = plt.subplots(
            len(self.meshes), len(self.controls), sharex=True, sharey=True
        )

        # Get minimum and maximum relative errors across all mesh/control
        # combinations so that data can be plotted on the same scale.
        # z_min, z_max = np.inf, -np.inf
        # for i, _ in enumerate(self.meshes):
        #     for j, _ in enumerate(self.controls):
        #         surrogate_errors = self.data[i][j]["surrogate_errors"]
        #         errors = self.data[i][j]["errors"]
        #         z = errors.mean(-1) / surrogate_errors.mean(-1)
        #         z_min = min(z_min, z.min())
        #         z_max = max(z_max, z.max())
        z_min = 1e-3
        z_max = 2e3

        # Add x-axis labels to bottommost row of plots
        for j, _ in enumerate(self.controls):
            # axs[-1, j].set_xlabel(r"noise $\epsilon$")
            axs[-1, j].set_xlabel("surrogate error")

        # Add y-axis labels to leftmost column of plots
        for i, mesh in enumerate(self.meshes):
            degree = mesh["degree"]
            num_elements = mesh["num_elements"]
            axs[i, 0].set_ylabel(f"(K={num_elements}, p={degree})\n" + r"gain $k$")

        # Add titles to topmost row of plots
        for j, control_name in enumerate(self.control_names):
            axs[0, j].set_title(control_name)

        # Create len(self.meshes)-by-len(self.controls) grid of contour plots
        levels = 10 ** np.linspace(np.log10(z_min), np.log10(z_max), 50)
        norm = LogTwoSlopeNorm(vcenter=1.0, vmin=z_min, vmax=z_max)

        for i, mesh in enumerate(self.meshes):
            for j, _ in enumerate(self.controls):
                ax = axs[i, j]

                expected_surrogate_error = (
                    final_surrogate_error_mean_div_noise * self.noises
                )
                relative_expected_surrogate_error = (
                    expected_surrogate_error / exact_solution_norm
                )
                x = relative_expected_surrogate_error

                y = self.gains

                x, y = np.meshgrid(x, y)

                x_min, x_max = x.min(), x.max()
                y_min, y_max = y.min(), y.max()

                ax.set_xscale("log")
                ax.set_yscale("log")

                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)

                surrogate_errors = self.data[i][j]["surrogate_errors"]
                errors = self.data[i][j]["errors"]

                # Take averages then compute relative error
                # z = errors.mean(-1) / surrogate_errors.mean(-1)

                # Compute relative error than take average
                z = (errors / surrogate_errors).mean(-1)

                cs = ax.contourf(x, y, z, cmap="seismic", norm=norm, levels=levels)

        # Add colorbar to right of plots
        cbar = fig.colorbar(cs, ax=axs, extend="both")
        cbar.locator = ticker.LogLocator(base=10.0, subs=(1.0,), numticks=10)
        cbar.update_ticks()
        cbar.ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10.0))

        # plt.tight_layout()
        plt.show()


class FigAdvec1dNudgingVsBaselinSchemeAndSurrogate(Figure):
    def __init__(self):
        super().__init__(
            name="advec1d_nuging_vs_baseline_scheme_and_surrogate", caption=""
        )

        self.meshes = [
            {"degree": 2, "num_elements": 15},
            {"degree": 3, "num_elements": 10},
            {"degree": 6, "num_elements": 4},
            {"degree": 10, "num_elements": 2},
        ]

        self.controls = [
            IdentityControl,
            ConservativeControl,
            ConservativeAndStableControl,
        ]
        self.control_names = ["Identity", "Conservative", "Conservative and stable"]

        self.num_samples = 10
        self.noises = 10 ** np.linspace(-4, 0, 50)
        self.gains = 10 ** np.linspace(-1, 2.8, 50)

        self.data = []

    def _get_default_filepath(self, filepath: Optional[str] = None) -> str:
        if filepath is None:
            return f"data/{self.name}_data.npz"

        if filepath.split(".")[-1] != "npz":
            filepath = f"{filepath}.npz"

        return filepath

    def _solve_and_compute_error(
        self,
        degree: int,
        num_elements: int,
        control: Optional[Control] = None,
        num_samples: Optional[int] = None,
    ) -> float:
        _, _, error = solve(
            xl,
            xr,
            advection_velocity,
            final_time,
            initial_condition,
            num_elements,
            degree,
            control,
            num_samples,
        )
        return error

    def save_data(self, filepath=None):
        filepath = self._get_default_filepath(filepath)
        np.savez(filepath, self.data)
        print(f"Saved data to {filepath}")

    def load_data(self, filepath=None):
        filepath = self._get_default_filepath(filepath)
        data = np.load(filepath, allow_pickle=True)["arr_0"]
        self.data = [
            [data[i][j] for j in range(len(self.controls))]
            for i in range(len(self.meshes))
        ]
        print(f"Loaded data from {filepath}")

    def generate_data(self):
        self.data = [
            [{} for j in range(len(self.controls))] for i in range(len(self.meshes))
        ]

        # Compute baseline errors
        for i, mesh in enumerate(self.meshes):
            num_elements = mesh["num_elements"]
            degree = mesh["degree"]
            baseline_error = self._solve_and_compute_error(degree, num_elements)
            for j in range(len(self.controls)):
                self.data[i][j]["baseline_error"] = float(baseline_error)

        # Re-use same randomly selected coefficients for all experiments
        coef = np.random.normal(size=(self.num_samples, 10, 10), loc=0, scale=1)

        for i, mesh in enumerate(self.meshes):
            degree = mesh["degree"]
            num_elements = mesh["num_elements"]
            for j, control_cls in enumerate(self.controls):
                # broadcast noises (epsilon) over samples, elements, and nodes
                noise = self.noises[:, None, None, None]

                # broadcast gains (k) over noises (epsilon), samples, elements, and nodes
                gain = self.gains[:, None, None, None, None]

                # Artificial surrogate
                surrogate = lambda x, t: (
                    exact_solution(x, t) + noise * randfun(x, t, coef)
                )
                control = control_cls(gain, surrogate)

                _, mesh, errors = solve(
                    xl,
                    xr,
                    advection_velocity,
                    final_time,
                    initial_condition,
                    num_elements=num_elements,
                    degree=degree,
                    control=control,
                    batch_shape=(len(self.gains), len(self.noises), self.num_samples),
                )

                assert errors.shape == (
                    len(self.gains),
                    len(self.noises),
                    self.num_samples,
                )
                self.data[i][j]["errors"] = errors

                # Compute surrogate errors
                P = mesh.J * mesh.ref_elem.P
                u_pred = exact_solution(mesh.x, final_time) + self.noises[
                    :, None, None, None
                ] * randfun(mesh.x, final_time, coef)
                u_exact = exact_solution(mesh.x, final_time)
                surrogate_errors = np.sqrt((P * (u_pred - u_exact) ** 2).sum((-2, -1)))
                self.data[i][j]["surrogate_errors"] = surrogate_errors

    def plot(self):
        fig, axs = plt.subplots(
            len(self.meshes), len(self.controls), sharex=True, sharey=True
        )

        # Get minimum and maximum relative errors across all mesh/control
        # combinations so that data can be plotted on the same scale.
        # z_min, z_max = np.inf, -np.inf
        # for i, _ in enumerate(self.meshes):
        #     for j, _ in enumerate(self.controls):
        #         baseline_error = self.data[i][j]["baseline_error"]
        #         surrogate_errors = self.data[i][j]["surrogate_errors"]
        #         errors = self.data[i][j]["errors"]
        #         z_baseline = errors.mean(-1) / baseline_error
        #         z_surrogate = errors.mean(-1) / surrogate_errors.mean(-1)
        #         z = np.maximum(z_baseline, z_surrogate)
        #         z_min = min(z_min, z.min())
        #         z_max = max(z_max, z.max())
        z_min = 1e-3
        z_max = 2e3

        # Add x-axis labels to bottommost row of plots
        for j, _ in enumerate(self.controls):
            # axs[-1, j].set_xlabel(r"noise $\epsilon$")
            axs[-1, j].set_xlabel("surrogate error")

        # Add y-axis labels to leftmost column of plots
        for i, mesh in enumerate(self.meshes):
            degree = mesh["degree"]
            num_elements = mesh["num_elements"]
            axs[i, 0].set_ylabel(f"(K={num_elements}, p={degree})\n" + r"gain $k$")

        # Add titles to topmost row of plots
        for j, control_name in enumerate(self.control_names):
            axs[0, j].set_title(control_name)

        # Create len(self.meshes)-by-len(self.controls) grid of contour plots
        levels = 10 ** np.linspace(np.log10(z_min), np.log10(z_max), 50)
        norm = LogTwoSlopeNorm(vcenter=1.0, vmin=z_min, vmax=z_max)

        for i, mesh in enumerate(self.meshes):
            for j, _ in enumerate(self.controls):
                ax = axs[i, j]

                expected_surrogate_error = (
                    final_surrogate_error_mean_div_noise * self.noises
                )
                relative_expected_surrogate_error = (
                    expected_surrogate_error / exact_solution_norm
                )
                x = relative_expected_surrogate_error

                y = self.gains

                x, y = np.meshgrid(x, y)

                x_min, x_max = x.min(), x.max()
                y_min, y_max = y.min(), y.max()

                ax.set_xscale("log")
                ax.set_yscale("log")

                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)

                baseline_error = self.data[i][j]["baseline_error"]
                surrogate_errors = self.data[i][j]["surrogate_errors"]
                errors = self.data[i][j]["errors"]
                z_baseline = errors.mean(-1) / baseline_error
                z_surrogate = (errors / surrogate_errors).mean(-1)

                z = np.maximum(z_baseline, z_surrogate)

                cs = ax.contourf(x, y, z, cmap="seismic", norm=norm, levels=levels)

        # Add colorbar to right of plots
        cbar = fig.colorbar(cs, ax=axs, extend="both")
        cbar.locator = ticker.LogLocator(base=10.0, subs=(1.0,), numticks=10)
        cbar.update_ticks()
        cbar.ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10.0))

        # plt.tight_layout()
        plt.show()


class FigAdvec1dNudgingErrorConvergence(Figure):
    def __init__(self, p: int, num_elements: Sequence[int]):
        super().__init__(name=f"advec1d_nuging_err_cv_p{p}_simpler_ic", caption="")

        self.degree = self.p = p

        self.controls = [
            IdentityControl,
            ConservativeControl,
            ConservativeAndStableControl,
        ]
        self.control_names = [
            "Weakly Stable",
            "Conservative",
            "Conservative & Energy-stable",
        ]

        self.num_samples = 10
        self.num_elements = np.array(num_elements)

        self.surrogate_errors = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1])
        _final_surrogate_error_mean_div_noise = 1.1432261796804175  # from monte carlo
        self.noises = self.surrogate_errors / _final_surrogate_error_mean_div_noise

        self.data = []

        N = 25
        coef = np.random.normal(scale=1, loc=0, size=(2, N))
        scaling_fn = lambda n: 1 / n
        scaling = scaling_fn(np.arange(1, N + 1))
        coef = coef * scaling

        sine_arg = np.array([2 * jnp.pi * n for n in range(N)])

        @jax.jit
        def initial_condition(x):
            y = (
                coef[0] * jnp.sin(sine_arg * x[..., None])
                + coef[1] * jnp.cos(sine_arg * x[..., None])
            ).sum(-1)
            return y

        self.initial_condition = initial_condition
        self.xl = 0
        self.xr = 1
        self.advection_velocity = 1
        self.final_time = 1
        self.exact_solution = lambda x, t: initial_condition(
            x - self.advection_velocity * t
        )

    def _get_default_filepath(self, filepath: Optional[str] = None) -> str:
        if filepath is None:
            return f"data/{self.name}_data.npz"

        if filepath.split(".")[-1] != "npz":
            filepath = f"{filepath}.npz"

        return filepath

    def save_data(self, filepath=None):
        filepath = self._get_default_filepath(filepath)
        np.savez(filepath, self.data)
        print(f"Saved data to {filepath}")

    def load_data(self, filepath=None):
        filepath = self._get_default_filepath(filepath)
        data = np.load(filepath, allow_pickle=True)["arr_0"].item()
        # self.data = [
        #     [data[i][j] for j in range(len(self.controls))] for i in range(len(self.meshes))
        # ]
        self.data = data
        print(f"Loaded data from {filepath}")

    def generate_data(self):
        self.data = {
            "baseline_errors": [],
            "nudging_errors": [[] for _ in range(len(self.controls))],
        }

        # Compute baseline errors
        for K in tqdm(self.num_elements):
            _, _, error = solve(
                self.xl,
                self.xr,
                self.advection_velocity,
                self.final_time,
                self.initial_condition,
                num_elements=K,
                degree=self.degree,
            )
            self.data["baseline_errors"].append(error)

        # Re-use same randomly selected coefficients for all experiments
        coef = np.random.normal(size=(self.num_samples, 10, 10), loc=0, scale=1)

        # Compute nudging errors
        for i, control_cls in enumerate(self.controls):
            # broadcast noises (epsilon) over samples, elements, and nodes
            noise = self.noises[:, None, None, None]

            # Artificial surrogate
            surrogate = lambda x, t: (
                self.exact_solution(x, t) + noise * randfun(x, t, coef)
            )

            for K in tqdm(self.num_elements):
                control = control_cls(None, surrogate)

                _, _, errors = solve(
                    self.xl,
                    self.xr,
                    self.advection_velocity,
                    self.final_time,
                    self.initial_condition,
                    num_elements=K,
                    degree=self.degree,
                    control=control,
                    batch_shape=(len(self.noises), self.num_samples),
                )
                assert errors.shape == (len(self.noises), self.num_samples)

                self.data["nudging_errors"][i].append(errors)

    def plot(self):
        fig, axs = plt.subplots(1, len(self.controls), sharex=True, sharey=True)

        # Add x-axis labels to bottommost row of plots
        for j, _ in enumerate(self.controls):
            axs[j].set_xlabel(r"h")

        # Add y-axis labels to leftmost column of plots
        axs[0].set_ylabel(r"$L^2$ error")

        # Add titles to topmost row of plots
        for j, control_name in enumerate(self.control_names):
            axs[j].set_title(control_name)

        hs = (self.xr - self.xl) / self.num_elements
        for j, _ in enumerate(self.controls):
            axs[j].set_xscale("log")
            axs[j].set_yscale("log")

            axs[j].plot(
                hs, self.data["baseline_errors"], label="baseline", c="black", ls="-"
            )

            errors = np.array(
                [
                    self.data["nudging_errors"][j][k]
                    for k in range(len(self.num_elements))
                ]
            )

            for i, surrogate_error in enumerate(self.surrogate_errors):
                axs[j].fill_between(
                    hs,
                    errors[:, i].min(-1),
                    errors[:, i].max(-1),
                    alpha=0.2,
                    color=f"C{i}",
                )
                axs[j].plot(
                    hs,
                    errors[:, i].mean(-1),
                    label=f"surrogate error = {surrogate_error}",
                    color=f"C{i}",
                )

            if j == len(self.controls) - 1:
                axs[j].legend()

        plt.show()


if __name__ == "__main__":
    fig = FigAdvec1dNudgingErrorConvergence(
        p=3, num_elements=[k for k in range(4, 251) if k % 7 == 0]
    )
    fig.generate_data()
    fig.save_data()
    fig.load_data()
    fig.plot()

    fig = FigAdvec1dNudgingVsBaselineScheme()
    fig.generate_data()
    fig.save_data()
    fig.load_data()
    fig.plot()

    fig = FigAdvec1dNudgingVsSurrogate()
    fig.generate_data()
    fig.save_data()
    fig.load_data()
    fig.plot()

    fig = FigAdvec1dNudgingVsBaselinSchemeAndSurrogate()
    fig.generate_data()
    fig.save_data()
    fig.load_data()
    fig.plot()
