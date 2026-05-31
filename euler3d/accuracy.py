import json

import matplotlib.pyplot as plt

from sbplite4py.utils.errors import get_least_squares_rate


def plot_data():

    with open("./data/euler3d_accuracy.json", "r") as f:
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
    plt.savefig("plots/euler3d_accuracy.jpg")


if __name__ == "__main__":
    plot_data()
