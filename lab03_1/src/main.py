import numpy as np
import matplotlib.pyplot as plt


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



def task_1(n: int = 500, h: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)

    k = np.arange(n + 1)
    x_clean = np.sqrt(k * h)
    x_noisy = x_clean + rng.normal(0, 1, n + 1)
    
    return x_noisy, x_clean


def task_2(n: int, x: np.ndarray, m: np.ndarray) -> list[np.ndarray]:
    data = []

    for m_val in m:
        y = np.full(n + 1, np.nan)
        for k in range(m_val, n - m_val + 1):
            y[k] = np.mean(x[k - m_val : k + m_val + 1])
        
        data.append(y)

    return data

    

def main() -> None:
    """ ЗАДАНИЕ 1"""
    n = 500
    x_noisy, x_clean = task_1()
    draw_plot(
        n,
        series=[
            (x_noisy, "noisy", dict(color="tab:orange", linestyle='None', marker='.', alpha=0.5)),
            (x_clean, "clean", dict(color="tab:blue",  linewidth=2.0)),
        ],
        title=r"Задание 1. $x_k = \sqrt{kh} + \xi_k$",
        xlabel="k",
        ylabel=r"$x_k$"
    )

    """ ЗАДАНИЕ 2"""
    m_windows = [10, 25, 55]
    ma_trends = task_2(n, x_noisy, m_windows)

    series_task_2 = [
        (x_noisy, "noisy", dict(color="lightgray", linewidth=0.6, alpha=0.6)),
        (x_clean, r"$\sqrt{kh}$ (точный тренд)", dict(color="black", linewidth=2.0, linestyle="--")),
    ]
    # добавляем три сглаженных ряда
    colors = ["tab:red", "tab:green", "tab:blue"]
    for m_val, trend, color in zip(m_windows, ma_trends, colors):
        w = 2 * m_val + 1  # фактическая ширина окна
        series_task_2.append(
            (trend, f"MA w={w}", dict(color=color, linewidth=1.6))
        )

    draw_plot(
        n,
        series=series_task_2,
        title=r"Задание 2. Простое скользящее среднее (окна 21, 51, 111)",
        xlabel="k",
        ylabel=r"$x_k$",
    )



if __name__ == "__main__":
    main()