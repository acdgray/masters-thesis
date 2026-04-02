import json
import matplotlib.pyplot as plt
from sbplite4py.utils.errors import get_least_squares_rate

from solver import solve, compute_error


def generate_data() -> None:
    final_time = 1
    degrees = [1, 2, 3, 4, 5]
    num_elements = [4, 8, 16, 32, 64]

    out = {}
    for degree in degrees:
        out[degree] = []
        for num_elements_ in num_elements:
            h = 1 / num_elements_
            solution, mesh = solve(final_time, degree, num_elements_)
            error = compute_error(solution, mesh, final_time)
            out[degree].append({"h": h, "error": error})

    with open("./data/advec2d_accuracy.json", "w") as f:
        f.write(json.dumps(out, indent=4))


def plot_data() -> None:

    fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=200)

    with open("./data/advec2d_accuracy.json", "r") as f:
        data = json.load(f)

    for p, v in data.items():
        x = [d["h"] for d in v]
        y = [d["error"] for d in v]
        rate = get_least_squares_rate(x[-4:], y[-4:])
        ax.scatter(x, y, label=f"p={p}, rate={rate:.2f}")
        ax.plot(x, y)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("h")
    ax.set_ylabel(r"$L^2$ error")

    plt.legend()
    plt.savefig("./plots/advec2d_accuracy.jpg")


if __name__ == "__main__":
    # generate_data()
    plot_data()
