import json

import matplotlib.pyplot as plt
from sbplite4py.utils.errors import get_least_squares_rate

from solver import solve, compute_error

schedule = {
    1: [10, 20, 30, 40, 50, 60, 70, 80],
    2: [10, 20, 30, 40, 50, 60, 70],
    3: [10, 20, 30, 40, 50, 60],
    4: [10, 20, 30, 40, 50],
}


def generate_data():

    errors = {1: [], 2: [], 3: [], 4: []}

    for p, Ns in schedule.items():
        for N in Ns:
            solution, mesh = solve(N, N, p, final_time=1, CFL=0.1, dissipation=False)
            error = compute_error(solution, mesh)
            errors[p].append({"h": 1 / N, "error": error})

    with open("data/accuracy.json", "w") as f:
        f.write(json.dumps(errors, indent=4))


def plot_data():

    with open("./data/accuracy.json", "r") as f:
        data = json.load(f)

    fig, axs = plt.subplots(1, 1, figsize=(6, 4), dpi=200)

    for p, v in data.items():
        x = [d["h"] for d in v]
        y = [d["error"] for d in v]
        rate = get_least_squares_rate(x[-4:], y[-4:])
        axs.scatter(x, y, label=f"p = {p}, rate={rate:.2f}")
        axs.plot(x, y)

    axs.set_xscale("log")
    axs.set_yscale("log")
    axs.set_xlabel(r"$h$")
    axs.set_ylabel(r"$L^2$-error")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/euler2d_accuracy.jpg")


if __name__ == "__main__":
    # generate_data()
    plot_data()
