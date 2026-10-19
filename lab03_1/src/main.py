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


# Вспомогательная: критерий Харке-Бера на нормальность (для задания 5)
def jarque_bera(r: np.ndarray) -> tuple[float, float, float]:
    n = len(r)
    d = r - r.mean()
    s2 = np.mean(d ** 2)
    skew = np.mean(d ** 3) / s2 ** 1.5
    kurt = np.mean(d ** 4) / s2 ** 2 - 3
    jb = n / 6 * (skew ** 2 + kurt ** 2 / 4)
    p_value = float(np.exp(-jb / 2))
    return float(skew), float(kurt), p_value


# Задание 1: модельный ряд
def task_1(n: int = 500, h: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)

    k = np.arange(n + 1)
    x_clean = 0.5 * np.sin(k * h)
    x_noisy = x_clean + rng.normal(0, 1, n + 1)

    return x_noisy, x_clean


# Задание 2: тренд экспоненциальным скользящим средним
def task_2(x: np.ndarray, alphas: list[float]) -> list[np.ndarray]:
    trends = []
    for alpha in alphas:
        y = np.empty_like(x)
        y[0] = x[:3].mean()
        for k in range(1, len(x)):
            y[k] = alpha * x[k] + (1 - alpha) * y[k - 1]
        trends.append(y)
    return trends


# Задание 3: сравнение трендов с точным значением
def task_3(
        x_clean: np.ndarray,
        alphas: list[float],
        trends: list[np.ndarray],
        ) -> list[dict]:
    rows = []
    for alpha, trend in zip(alphas, trends):
        err = trend - x_clean
        rows.append(dict(
            alpha=alpha,
            m=round(1 / alpha - 1),
            bias=float(err.mean()),
            rmse=float(np.sqrt(np.mean(err ** 2))),
            max_abs=float(np.abs(err).max()),
            amplitude=float((trend.max() - trend.min()) / 2),
        ))
    return rows


# Задание 4: амплитудный спектр Фурье
def task_4(x: np.ndarray, h: float) -> dict:
    n = len(x)
    amplitude = np.abs(np.fft.rfft(x))
    freq = np.arange(len(amplitude)) / (n * h)

    j_main = int(np.argmax(amplitude[1:])) + 1
    return dict(
        freq=freq,
        amplitude=amplitude,
        j_main=j_main,
        f_main=float(freq[j_main]),
        omega_main=float(2 * np.pi * freq[j_main]),
        period=float(1 / freq[j_main]),
    )


# Задание 5: проверка остатков (случайность, несмещенность, нормальность)
def task_5(x: np.ndarray, named_trends: list[tuple[str, np.ndarray]]) -> list[dict]:
    rows = []
    for name, trend in named_trends:
        r = x - trend
        n = len(r)
        p, e_p, s_p = turning_points(r)
        tau, s_tau = kendall_tau(r)
        t_mean = r.mean() / (r.std(ddof=1) / np.sqrt(n))
        skew, kurt, p_norm = jarque_bera(r)
        rows.append(dict(
            name=name, residual=r,
            p=p, e_p=e_p, s_p=s_p, z_p=(p - e_p) / s_p,
            tau=tau, s_tau=s_tau, z_tau=tau / s_tau,
            random=abs(p - e_p) < 2 * s_p and abs(tau) < 2 * s_tau,
            mean=float(r.mean()), std=float(r.std(ddof=1)), t_mean=float(t_mean),
            unbiased=abs(t_mean) < 2,
            skew=skew, kurt=kurt, p_norm=p_norm,
            normal=p_norm > 0.05,
        ))
    return rows


# Вывод таблицы задания 3
def print_task_3(rows: list[dict]) -> None:
    print(f"{'alpha':>6} {'m':>4} {'смещение':>9} {'RMSE':>7} {'max|ош.|':>9} {'размах/2':>9}")
    for r in rows:
        print(f"{r['alpha']:>6} {r['m']:>4} {r['bias']:>+9.3f} {r['rmse']:>7.3f} "
              f"{r['max_abs']:>9.3f} {r['amplitude']:>9.3f}")
    print()


# Вывод результата задания 4
def print_task_4(res: dict) -> None:
    print(f"главная гармоника: номер j = {res['j_main']}, амплитуда A = {res['amplitude'][res['j_main']]:.1f}")
    print(f"частота f = {res['f_main']:.4f}, угловая частота w = {res['omega_main']:.4f}, "
          f"период T = {res['period']:.2f}")
    print("точные значения модельного ряда: w = 1, f = 1/(2*pi) = 0.1592, T = 2*pi = 6.28")
    print()


# Вывод таблицы задания 5
def print_task_5(rows: list[dict]) -> None:
    print("случайность: |p - E(p)| < 2 s(p) и |tau| < 2 s(tau)")
    print(f"{'остаток':<14} {'p':>4} {'E(p)':>7} {'z_p':>6} {'tau':>7} {'z_tau':>6} {'случаен':>8}")
    for r in rows:
        print(f"{r['name']:<14} {r['p']:>4} {r['e_p']:>7.1f} {r['z_p']:>+6.2f} "
              f"{r['tau']:>+7.3f} {r['z_tau']:>+6.2f} {'да' if r['random'] else 'НЕТ':>8}")
    print()
    print("несмещенность: |t| < 2, нормальность: p-value критерия Харке-Бера > 0.05")
    print(f"{'остаток':<14} {'среднее':>8} {'СКО':>6} {'t':>6} {'несмещ.':>8} "
          f"| {'асимм.':>7} {'эксцесс':>8} {'p-value':>8} {'нормален':>9}")
    for r in rows:
        print(f"{r['name']:<14} {r['mean']:>+8.3f} {r['std']:>6.3f} {r['t_mean']:>+6.2f} "
              f"{'да' if r['unbiased'] else 'НЕТ':>8} "
              f"| {r['skew']:>+7.3f} {r['kurt']:>+8.3f} {r['p_norm']:>8.3f} "
              f"{'да' if r['normal'] else 'НЕТ':>9}")


# Вспомогательная: график спектра (для задания 4)
def draw_spectrum(res: dict, j_max: int = 60) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=120)
    ax.plot(res["freq"][: j_max + 1], res["amplitude"][: j_max + 1],
            color="tab:blue", linewidth=1.4, marker=".")
    jm = res["j_main"]
    ax.plot(res["f_main"], res["amplitude"][jm], "o", color="tab:red", markersize=9,
            label=f"главная частота f = {res['f_main']:.4f} (w = {res['omega_main']:.3f})")
    ax.axvline(1 / (2 * np.pi), color="black", linestyle="--", linewidth=1,
               label=r"точное значение $1/(2\pi)$")

    ax.set_title("Задание 4. Амплитудный спектр Фурье ряда $x_k$", fontsize=13, pad=12)
    ax.set_xlabel("частота f (циклов на единицу t = kh)", fontsize=12)
    ax.set_ylabel("амплитуда A(f)", fontsize=12)
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()


# Вспомогательная: гистограммы остатков (для задания 5)
def draw_histograms(rows: list[dict]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), dpi=110)
    for ax, r in zip(axes.ravel(), rows):
        res = r["residual"]
        ax.hist(res, bins=20, density=True, color="lightblue", edgecolor="gray")
        t = np.linspace(res.min(), res.max(), 200)
        ax.plot(t, np.exp(-(t - res.mean()) ** 2 / (2 * res.std() ** 2))
                / (res.std() * np.sqrt(2 * np.pi)), color="tab:red", linewidth=1.5)
        ax.set_title(f"{r['name']}\nсреднее={r['mean']:+.3f}, СКО={r['std']:.3f}, "
                     f"p={r['p_norm']:.3f}", fontsize=10)
        ax.set_xlabel("остаток")
        ax.set_ylabel("плотность")
    for ax in axes.ravel()[len(rows):]:
        ax.axis("off")
    fig.suptitle("Задание 5. Гистограммы остатков и нормальная плотность", fontsize=13)
    fig.tight_layout()


def main() -> None:
    n = 500
    h = 0.1
    alphas = [0.01, 0.05, 0.1, 0.3]
    x_noisy, x_clean = task_1(n, h)

    # Задание 1
    draw_plot(
        n,
        series=[
            (x_noisy, "noisy", dict(color="tab:orange", linestyle="None", marker=".", alpha=0.5)),
            (x_clean, "clean", dict(color="tab:blue", linewidth=2.0)),
        ],
        title=r"Задание 1. $x_k = 0.5\sin(kh) + \xi_k$",
        xlabel="k",
        ylabel=r"$x_k$",
    )

    # Задание 2
    trends = task_2(x_noisy, alphas)
    colors = ["tab:red", "tab:green", "tab:purple", "tab:brown"]
    series = [
        (x_noisy, "noisy", dict(color="lightgray", linewidth=0.6, alpha=0.7)),
        (x_clean, r"$0.5\sin(kh)$ (точный тренд)", dict(color="black", linewidth=2.0, linestyle="--")),
    ]
    for alpha, trend, color in zip(alphas, trends, colors):
        series.append((trend, rf"EMA $\alpha$={alpha}", dict(color=color, linewidth=1.6)))
    draw_plot(
        n,
        series=series,
        title="Задание 2. Экспоненциальное скользящее среднее",
        xlabel="k",
        ylabel=r"$x_k$",
    )

    # Задание 3
    print("=" * 30, "ЗАДАНИЕ 3", "=" * 30)
    print_task_3(task_3(x_clean, alphas, trends))
    draw_plot(
        n,
        series=[(t - x_clean, rf"EMA $\alpha$={a}", dict(color=c, linewidth=1.3))
                for a, t, c in zip(alphas, trends, colors)],
        title=r"Задание 3. Ошибка тренда относительно $0.5\sin(kh)$",
        xlabel="k",
        ylabel="тренд − точное значение",
    )

    # Задание 4
    print("=" * 30, "ЗАДАНИЕ 4", "=" * 30)
    spectrum = task_4(x_noisy, h)
    print_task_4(spectrum)
    draw_spectrum(spectrum)

    # Задание 5
    print("=" * 30, "ЗАДАНИЕ 5", "=" * 30)
    print(f"n = {n + 1} точек")
    named = [("шум (эталон)", x_clean)]
    named += [(f"EMA a={a}", t) for a, t in zip(alphas, trends)]
    rows = task_5(x_noisy, named)
    print_task_5(rows)
    draw_histograms(rows)

    plt.show()


if __name__ == "__main__":
    main()