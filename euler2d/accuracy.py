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

    data = {}
    for flag, label in [(False, "EC"), (True, "ES")]:
        errors = {1: [], 2: [], 3: [], 4: []}

        for p, Ns in schedule.items():
            for N in Ns:
                solution, mesh = solve(
                    N, N, p, final_time=1, CFL=0.01, dissipation=flag
                )
                error = compute_error(solution, mesh)
                errors[p].append({"h": 1 / N, "error": error})

        data[label] = errors

    with open("data/euler2d_accuracy.json", "w") as f:
        f.write(json.dumps(data, indent=4))


def plot_data():

    with open("./data/euler2d_accuracy.json", "r") as f:
        data = json.load(f)
        ec_data = data["EC"]
        es_data = data["ES"]

    fig, axs = plt.subplots(1, 2, figsize=(10, 4), dpi=200)

    for p, v in ec_data.items():
        x = [d["h"] for d in v]
        y = [d["error"] for d in v]
        rate = get_least_squares_rate(x[-4:], y[-4:])
        axs[0].scatter(x, y, label=f"p = {p}, rate={rate:.2f}")
        axs[0].plot(x, y)

    for p, v in es_data.items():
        x = [d["h"] for d in v]
        y = [d["error"] for d in v]
        rate = get_least_squares_rate(x[-4:], y[-4:])
        axs[1].scatter(x, y, label=f"p = {p}, rate={rate:.2f}")
        axs[1].plot(x, y)

    axs[0].set_title("Entropy-conservative")
    axs[1].set_title("Entropy-stable")

    for i in range(2):
        axs[i].set_xscale("log")
        axs[i].set_yscale("log")
        axs[i].set_xlabel(r"$h$")
        axs[i].set_ylabel(r"$L^2$-error")
        axs[i].legend()

    plt.tight_layout()
    plt.savefig("plots/euler2d_accuracy.jpg")


if __name__ == "__main__":
    # generate_data()
    plot_data()
