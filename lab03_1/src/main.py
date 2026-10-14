from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np


def draw_plot(
        n: int,
        series: list[tuple[np.ndarray, str, dict]],
        title: str,
        xlabel: str,
        ylabel: str,
        ) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=120)
    k = np.arange(n + 1)
    for y, label, style in series:
        ax.plot(k, y, label=label, **style)

    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    plt.show()


def moving_window(
        n: int,
        x: np.ndarray,
        m_list: list[int],
        func: Callable[[np.ndarray], float],
        ) -> list[np.ndarray]:
    result = []
    for m in m_list:
        y = np.full(n + 1, np.nan)
        for k in range(m, n - m + 1):
            y[k] = func(x[k - m : k + m + 1])
        result.append(y)
    return result


def fill_edges_poly(x: np.ndarray, y: np.ndarray, m: int, degree: int = 3) -> None:
    n = len(x) - 1
    w = 2 * m + 1
    t = np.arange(w)

    coef = np.polyfit(t, x[:w], degree)
    y[:m] = np.polyval(coef, np.arange(m))

    coef = np.polyfit(t, x[n - w + 1 :], degree)
    y[n - m + 1 :] = np.polyval(coef, np.arange(m + 1, w))
    

def task_1(n: int = 500, h: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)

    k = np.arange(n + 1)
    x_clean = np.sqrt(k * h)
    x_noisy = x_clean + rng.normal(0, 1, n + 1)

    return x_noisy, x_clean


def task_2(n: int, x: np.ndarray, m_list: list[int]) -> list[np.ndarray]:
    trends = moving_window(n, x, m_list, np.mean)
    for m, y in zip(m_list, trends):
        fill_edges_poly(x, y, m)
    return trends


def main() -> None:
    n = 500
    x_noisy, x_clean = task_1(n)

    # Задание 1
    draw_plot(
        n,
        series=[
            (x_noisy, "noisy", dict(color="tab:orange", linestyle="None", marker=".", alpha=0.5)),
            (x_clean, "clean", dict(color="tab:blue", linewidth=2.0)),
        ],
        title=r"Задание 1. $x_k = \sqrt{kh} + \xi_k$",
        xlabel="k",
        ylabel=r"$x_k$",
    )

    # Задание 2
    m_windows = [10, 25, 55]
    ma_trends = task_2(n, x_noisy, m_windows)

    series_task_2 = [
        (x_noisy, "noisy", dict(color="lightgray", linewidth=0.6, alpha=0.6)),
        (x_clean, r"$\sqrt{kh}$ (точный тренд)", dict(color="black", linewidth=2.0, linestyle="--")),
    ]
    colors = ["tab:red", "tab:green", "tab:blue"]
    for m, trend, color in zip(m_windows, ma_trends, colors):
        series_task_2.append((trend, f"MA w={2 * m + 1}", dict(color=color, linewidth=1.6)))

    draw_plot(
        n,
        series=series_task_2,
        title="Задание 2. Простое скользящее среднее (окна 21, 51, 111)",
        xlabel="k",
        ylabel=r"$x_k$",
    )


if __name__ == "__main__":
    main()