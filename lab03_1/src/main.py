from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np


# Вспомогательная: построение графика
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
    # plt.show()


# Вспомогательная: скользящее окно (для заданий 2 и 3)
def moving_window(
        n: int,
        x: np.ndarray,
        m_list: list[int],
        func: Callable[[np.ndarray], float],
        tukey_edges: bool = False,
        ) -> list[np.ndarray]:
    result = []
    for m in m_list:
        y = np.full(n + 1, np.nan)

        first, last = (1, n - 1) if tukey_edges else (m, n - m)
        for k in range(first, last + 1):
            half = min(m, k, n - k)
            y[k] = func(x[k - half : k + half + 1])

        if tukey_edges:
            y[0] = func(np.array([x[0], x[1], 3 * x[1] - 2 * x[2]]))
            y[n] = func(np.array([x[n], x[n - 1], 3 * x[n - 1] - 2 * x[n - 2]]))

        result.append(y)
    return result


# Вспомогательная: края скользящего среднего (для задания 2)
def fill_edges_poly(x: np.ndarray, y: np.ndarray, m: int, degree: int = 3) -> None:
    n = len(x) - 1
    w = 2 * m + 1
    t = np.arange(w)

    coef = np.polyfit(t, x[:w], degree)
    y[:m] = np.polyval(coef, np.arange(m))

    coef = np.polyfit(t, x[n - w + 1 :], degree)
    y[n - m + 1 :] = np.polyval(coef, np.arange(m + 1, w))


# Вспомогательная: ряды для графиков заданий 2 и 3
def build_series(
        x_noisy: np.ndarray,
        x_clean: np.ndarray,
        m_list: list[int],
        trends: list[np.ndarray],
        prefix: str,
        ) -> list[tuple[np.ndarray, str, dict]]:
    series = [
        (x_noisy, "noisy", dict(color="lightgray", linewidth=0.6, alpha=0.6)),
        (x_clean, r"$\sqrt{kh}$ (точный тренд)", dict(color="black", linewidth=2.0, linestyle="--")),
    ]
    colors = ["tab:red", "tab:green", "tab:blue"]
    for m, trend, color in zip(m_list, trends, colors):
        series.append((trend, f"{prefix} w={2 * m + 1}", dict(color=color, linewidth=1.6)))
    return series


# Вспомогательная: число поворотных точек (для задания 5)
def turning_points(r: np.ndarray) -> tuple[int, float, float]:
    n = len(r)
    a, b, c = r[:-2], r[1:-1], r[2:]
    is_turn = ((a < b) & (b > c)) | ((a > b) & (b < c))
    p = int(is_turn.sum())
    e_p = 2 / 3 * (n - 2)
    s_p = np.sqrt((16 * n - 29) / 90)
    return p, e_p, s_p


# Вспомогательная: коэффициент Кендэла (для задания 5)
def kendall_tau(r: np.ndarray) -> tuple[float, float]:
    n = len(r)
    big_p = sum(int((r[i + 1 :] > r[i]).sum()) for i in range(n - 1))
    tau = 4 * big_p / (n * (n - 1)) - 1
    s_tau = np.sqrt(2 * (2 * n + 5) / (9 * n * (n - 1)))
    return tau, s_tau


# Задание 1: модельный ряд
def task_1(n: int = 500, h: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)

    k = np.arange(n + 1)
    x_clean = np.sqrt(k * h)
    x_noisy = x_clean + rng.normal(0, 1, n + 1)

    return x_noisy, x_clean


# Задание 2: тренд простым скользящим средним
def task_2(n: int, x: np.ndarray, m_list: list[int]) -> list[np.ndarray]:
    trends = moving_window(n, x, m_list, np.mean)
    for m, y in zip(m_list, trends):
        fill_edges_poly(x, y, m)
    return trends


# Задание 3: тренд скользящей медианой
def task_3(n: int, x: np.ndarray, m_list: list[int]) -> list[np.ndarray]:
    return moving_window(n, x, m_list, np.median, tukey_edges=True)


# Задание 4: сравнение трендов с точным значением
def task_4(
        x_clean: np.ndarray,
        m_list: list[int],
        trends: list[np.ndarray],
        ) -> list[dict]:
    n = len(x_clean) - 1
    rows = []
    for m, trend in zip(m_list, trends):
        err = trend - x_clean
        mid = slice(m, n - m + 1)
        edge = np.r_[0:m, n - m + 1 : n + 1]
        rows.append(dict(
            w=2 * m + 1,
            bias=float(err.mean()),
            rmse=float(np.sqrt(np.mean(err ** 2))),
            rmse_mid=float(np.sqrt(np.mean(err[mid] ** 2))),
            rmse_edge=float(np.sqrt(np.mean(err[edge] ** 2))),
            max_abs=float(np.abs(err).max()),
        ))
    return rows


# Задание 5: проверка остатков на случайность
def task_5(x: np.ndarray, named_trends: list[tuple[str, np.ndarray]]) -> list[dict]:
    rows = []
    for name, trend in named_trends:
        r = x - trend
        p, e_p, s_p = turning_points(r)
        tau, s_tau = kendall_tau(r)
        rows.append(dict(
            name=name, p=p, e_p=e_p, s_p=s_p, z_p=(p - e_p) / s_p,
            tau=tau, s_tau=s_tau, z_tau=tau / s_tau,
            random_p=abs(p - e_p) < 2 * s_p,
            random_tau=abs(tau) < 2 * s_tau,
        ))
    return rows


# Вывод таблицы задания 4
def print_task_4(title: str, rows: list[dict]) -> None:
    print(title)
    print(f"{'окно':>5} {'смещение':>9} {'RMSE':>7} {'RMSE сер.':>10} {'RMSE края':>10} {'max|ош.|':>9}")
    for r in rows:
        print(f"{r['w']:>5} {r['bias']:>+9.3f} {r['rmse']:>7.3f} {r['rmse_mid']:>10.3f} "
              f"{r['rmse_edge']:>10.3f} {r['max_abs']:>9.3f}")
    print()


# Вывод таблицы задания 5
def print_task_5(rows: list[dict]) -> None:
    print(f"{'остаток':<14} {'p':>4} {'E(p)':>7} {'s(p)':>6} {'z_p':>6} {'случаен':>8} "
          f"| {'tau':>7} {'s(tau)':>7} {'z_tau':>6} {'случаен':>8}")
    for r in rows:
        print(f"{r['name']:<14} {r['p']:>4} {r['e_p']:>7.1f} {r['s_p']:>6.2f} {r['z_p']:>+6.2f} "
              f"{'да' if r['random_p'] else 'НЕТ':>8} "
              f"| {r['tau']:>+7.3f} {r['s_tau']:>7.4f} {r['z_tau']:>+6.2f} "
              f"{'да' if r['random_tau'] else 'НЕТ':>8}")


def main() -> None:
    n = 500
    m_windows = [10, 25, 55]
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
    ma_trends = task_2(n, x_noisy, m_windows)
    draw_plot(
        n,
        series=build_series(x_noisy, x_clean, m_windows, ma_trends, "MA"),
        title="Задание 2. Простое скользящее среднее (окна 21, 51, 111)",
        xlabel="k",
        ylabel=r"$x_k$",
    )

    # Задание 3
    med_trends = task_3(n, x_noisy, m_windows)
    draw_plot(
        n,
        series=build_series(x_noisy, x_clean, m_windows, med_trends, "MED"),
        title="Задание 3. Скользящая медиана (окна 21, 51, 111)",
        xlabel="k",
        ylabel=r"$x_k$",
    )

    # Задание 4
    print("=" * 30, "ЗАДАНИЕ 4", "=" * 30)
    print_task_4("Скользящее среднее: ошибка относительно sqrt(kh)",
                 task_4(x_clean, m_windows, ma_trends))
    print_task_4("Скользящая медиана: ошибка относительно sqrt(kh)",
                 task_4(x_clean, m_windows, med_trends))

    colors = ["tab:red", "tab:green", "tab:blue"]
    for trends, prefix, name in ((ma_trends, "MA", "среднее"), (med_trends, "MED", "медиана")):
        draw_plot(
            n,
            series=[(t - x_clean, f"{prefix} w={2 * m + 1}", dict(color=c, linewidth=1.2))
                    for m, t, c in zip(m_windows, trends, colors)],
            title=f"Задание 4. Ошибка тренда (скользящая {name}) относительно $\\sqrt{{kh}}$",
            xlabel="k",
            ylabel="тренд − точное значение",
        )

    # Задание 5
    print("=" * 30, "ЗАДАНИЕ 5", "=" * 30)
    print(f"n = {n + 1} точек; критерий: |отклонение| < 2 sigma")
    named = [("шум (эталон)", x_clean)]
    named += [(f"MA w={2 * m + 1}", t) for m, t in zip(m_windows, ma_trends)]
    named += [(f"MED w={2 * m + 1}", t) for m, t in zip(m_windows, med_trends)]
    print_task_5(task_5(x_noisy, named))

    plt.show()
if __name__ == "__main__":
    main()