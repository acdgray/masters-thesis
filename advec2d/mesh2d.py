import jax
import matplotlib.pyplot as plt
import numpy as np

# Specify double precision floats before importing SBPLite4py
jax.config.update("jax_enable_x64", True)

from sbplite4py.mesh import CurvilinearMesh2D  # noqa: E402
from sbplite4py.ref_elem import LegendreGaussLobatto2D  # noqa: E402
from sbplite4py.curvilinear.transformations import hicken_et_al  # noqa: E402


def plot_element_edges(num_elements: int) -> None:
    degree = 20
    ref_elem = LegendreGaussLobatto2D(degree)
    vx = vy = np.linspace(0, 1, num_elements + 1)
    mesh = CurvilinearMesh2D(vx, vy, ref_elem, hicken_et_al)

    x = mesh.xy[..., 0]
    y = mesh.xy[..., 1]

    c = "black"
    zorder = 1

    for i in range(num_elements):
        for j in range(num_elements):
            for l in range(x.shape[2] - 1):
                plt.plot(
                    x[i, j, l : l + 2, -1], y[i, j, l : l + 2, -1], c=c, zorder=zorder
                )
                plt.plot(
                    x[i, j, l : l + 2, 0], y[i, j, l : l + 2, 0], c=c, zorder=zorder
                )
                plt.plot(
                    x[i, j, -1, l : l + 2], y[i, j, -1, l : l + 2], c=c, zorder=zorder
                )
                plt.plot(
                    x[i, j, 0, l : l + 2], y[i, j, 0, l : l + 2], c=c, zorder=zorder
                )


def plot_element_nodes(num_elements: int, degree: int) -> None:
    ref_elem = LegendreGaussLobatto2D(degree)
    vx = vy = np.linspace(0, 1, num_elements + 1)
    mesh = CurvilinearMesh2D(vx, vy, ref_elem, hicken_et_al)

    x = mesh.xy[..., 0]
    y = mesh.xy[..., 1]

    zorder = 2

    plt.scatter(x, y, edgecolors="black", facecolors="white", marker="o", zorder=zorder)


def plot_mesh(num_elements: int, degree: int) -> None:
    plot_element_edges(num_elements)
    plot_element_nodes(num_elements, degree)


plot_mesh(5, 3)

plt.gca().set_aspect("equal")
plt.savefig("plots/advec2d_mesh.png", dpi=500)
