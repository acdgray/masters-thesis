import json

import matplotlib.pyplot as plt

from solver import solve


def generate_data() -> None:

    final_time = 1
    degree = 3
    num_elements = 16

    solution, mesh = solve(final_time, degree, num_elements, alpha=0.5)

    ts = [float(t) for t in solution.ts[1]]
    mass = [float(x) for x in solution.ys[1]["mass"]]
    mass_rate = [float(x) for x in solution.ys[1]["mass_rate"]]

    data = {"ts": ts, "mass": mass, "mass_rate": mass_rate}

    with open("./data/advec2d_mass_conservation.json", "w") as f:
        f.write(json.dumps(data, indent=4))


def plot_data() -> None:
    with open("./data/advec2d_mass_conservation.json", "r") as f:
        data = json.load(f)

    ts = data["ts"]
    mass = data["mass"]
    mass_rate = data["mass_rate"]

    fig, axs = plt.subplots(1, 2, figsize=(10, 4), dpi=300)

    axs[0].plot(ts, mass_rate)
    axs[0].set_title("Mass rate")
    axs[0].set_xlabel("t")

    axs[1].plot(ts, mass)
    axs[1].set_title("Mass")
    axs[1].set_xlabel("t")

    plt.tight_layout()
    plt.savefig("./plots/advec2d_mass_conservation.png")


if __name__ == "__main__":
    # generate_data()
    plot_data()
