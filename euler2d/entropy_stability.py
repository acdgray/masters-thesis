import json

import matplotlib.pyplot as plt

from solver import solve


def generate_data():
    data = {}

    for flag, label in [(False, "entropy-conservative"), (True, "entropy-stable")]:
        solution, _ = solve(
            Nx=40, Ny=40, degree=3, final_time=1, CFL=0.01, dissipation=flag
        )
        data[label] = {
            "ts": [float(t) for t in solution.ts[1]],
            "entropy": [float(x) for x in solution.ys[1]["total_s"]],
            "entropy_rate": [float(x) for x in solution.ys[1]["total_ds"]],
        }

    with open("./data/euler2d_entropy_stability.json", "w") as f:
        f.write(json.dumps(data, indent=4))


def plot_data() -> None:
    with open("./data/euler2d_entropy_stability.json", "r") as f:
        data = json.load(f)

    ec_ts = data["entropy-conservative"]["ts"]
    es_ts = data["entropy-stable"]["ts"]
    ec_entropy = data["entropy-conservative"]["entropy"]
    ec_entropy_rate = data["entropy-conservative"]["entropy_rate"]
    es_entropy = data["entropy-stable"]["entropy"]
    es_entropy_rate = data["entropy-stable"]["entropy_rate"]

    fig, axs = plt.subplots(1, 2, figsize=(10, 4), dpi=300)

    axs[0].plot(ec_ts, ec_entropy_rate, label="EC")
    axs[0].plot(es_ts, es_entropy_rate, label="ES", ls="--")
    axs[0].set_title("Entropy rate")
    axs[0].set_xlabel("t")
    axs[0].legend()

    axs[1].plot(ec_ts, ec_entropy, label="EC")
    axs[1].plot(es_ts, es_entropy, label="ES", ls="--")
    axs[1].set_title("Entropy")
    axs[1].set_xlabel("t")
    axs[1].legend()

    plt.tight_layout()
    plt.savefig("./plots/euler2d_entropy_stability.png")


if __name__ == "__main__":
    # generate_data()
    plot_data()
