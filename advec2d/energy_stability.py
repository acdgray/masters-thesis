import json

import matplotlib.pyplot as plt

from solver import solve


def generate_data() -> None:
    final_time = 1
    degree = 3
    num_elements = 16
    alphas = [0, 0.5, 1.0]

    data = {}
    for alpha in alphas:
        solution, _ = solve(final_time, degree, num_elements, alpha)

        ts = solution.ts[1]
        energy = solution.ys[1]["energy"]
        energy_rate = solution.ys[1]["energy_rate"]

        data[alpha] = {
            "ts": [float(t) for t in ts],
            "Es": [float(x) for x in energy],
            "dEdts": [float(x) for x in energy_rate],
        }

    with open("./data/advec2d_energy_stability.json", "w") as f:
        f.write(json.dumps(data, indent=4))


def plot_data() -> None:

    with open("./data/advec2d_energy_stability.json", "r") as f:
        data = json.load(f)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4), dpi=300)
    axs[0].set_title("Energy rate")
    axs[0].set_xlabel("t")
    axs[1].set_title("Energy")
    axs[1].set_xlabel("t")

    for alpha, d in data.items():
        t = d["ts"]
        energy = d["Es"]
        energy_rate = d["dEdts"]
        label = r"$\alpha$" + "=" + str(alpha)
        axs[0].plot(t, energy_rate, label=label)
        axs[1].plot(t, energy, label=label)

    axs[1].legend()
    plt.tight_layout()
    plt.savefig("plots/advec2d_energy_stability.png")


if __name__ == "__main__":
    # generate_data()
    plot_data()
