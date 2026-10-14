import numpy as np
import matplotlib.pyplot as plt


def task_1(n: int = 500, h: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)

    k = np.arange(n + 1)
    x_clean = np.sqrt(k * h)
    x_noisy = x_clean + rng.normal(0, 1, n + 1)
    
    return x_noisy, x_clean


def main() -> None:
    n = 500
    h = 0.05
    x_noisy, x_clean = task_1(n, h)
    k = np.arange(n + 1)


    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=120)

    ax.scatter(
        k, x_noisy,
        s=10, alpha=0.5, color="tab:orange",
        label=r"$x_k^{\text{noisy}} = \sqrt{k h} + \xi_k$",
        zorder=2,
    )

    ax.plot(
        k, x_clean,
        color="tab:blue", linewidth=1.8,
        label=r"$x_k^{\text{clean}} = \sqrt{k h}$",
        zorder=3,
    )

    ax.set_title(r"Задание 1. Сигнал $x_k = \sqrt{k h}$ с шумом $\xi_k \sim \mathcal{N}(0,1)$",
                 fontsize=13, pad=12)
    ax.set_xlabel(r"$k$", fontsize=12)
    ax.set_ylabel(r"$x_k$", fontsize=12)

    # Легенда
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=11)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()