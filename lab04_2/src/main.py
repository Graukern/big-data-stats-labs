import matplotlib.pyplot as plt
import numpy as np


# Вспомогательная: правило трех сигм для одного значения (для задания 1)
def three_sigma_rule(value: float, mean: float, std: float) -> bool:
    return abs(value - mean) > 3 * std


# Вспомогательная: пороги боксплота Тьюки (для задания 2)
def tukey_fences(x: np.ndarray) -> tuple[float, float, float, float, float]:
    lq, med, uq = np.percentile(x, [25, 50, 75])
    iqr = uq - lq
    x_low = max(x.min(), lq - 1.5 * iqr)
    x_up = min(x.max(), uq + 1.5 * iqr)
    return lq, med, uq, x_low, x_up


# Вспомогательная: модельный ряд (для заданий 1 и 2)
def generate_series(rng: np.random.Generator, n: int = 195) -> np.ndarray:
    return np.append(rng.normal(0, 1, n), [5, -4, 3.3, 2.99, -3])


# Задание 1: правило трех сигм для трех первых и трех последних порядковых статистик
def task_1(x: np.ndarray) -> dict:
    mean = float(x.mean())
    std = float(x.std(ddof=1))
    n = len(x)

    rows = []
    for i in [0, 1, 2, n - 3, n - 2, n - 1]:
        rows.append(dict(
            number=i + 1,
            value=float(x[i]),
            z=abs(float(x[i]) - mean) / std,
            anomaly=three_sigma_rule(x[i], mean, std),
        ))
    return dict(mean=mean, std=std, rows=rows)


# Задание 2: боксплот Тьюки и сравнение с правилом трех сигм
def task_2(x: np.ndarray, rule: dict) -> dict:
    lq, med, uq, x_low, x_up = tukey_fences(x)
    outliers = x[(x < x_low) | (x > x_up)]

    sigma_outliers = x[np.abs(x - rule["mean"]) > 3 * rule["std"]]
    return dict(
        lq=lq, med=med, uq=uq, iqr=uq - lq, x_low=x_low, x_up=x_up,
        outliers=outliers,
        sigma_outliers=sigma_outliers,
        only_tukey=np.setdiff1d(outliers, sigma_outliers),
    )


# Дополнение к заданию 2: частота обнаружения при многократном повторении
def estimate_rates(n_runs: int = 2000, n: int = 195) -> dict:
    rng = np.random.default_rng(0)
    found_sigma = np.zeros(5)
    found_tukey = np.zeros(5)
    false_sigma = 0
    false_tukey = 0

    for _ in range(n_runs):
        x = generate_series(rng, n)
        by_sigma = np.abs(x - x.mean()) > 3 * x.std(ddof=1)
        _, _, _, x_low, x_up = tukey_fences(x)
        by_tukey = (x < x_low) | (x > x_up)

        found_sigma += by_sigma[n:]
        found_tukey += by_tukey[n:]
        false_sigma += by_sigma[:n].sum()
        false_tukey += by_tukey[:n].sum()

    return dict(
        n_runs=n_runs,
        sigma=found_sigma / n_runs,
        tukey=found_tukey / n_runs,
        false_sigma=false_sigma / n_runs,
        false_tukey=false_tukey / n_runs,
    )


# Вывод результатов дополнения
def print_rates(rates: dict) -> None:
    print(f"доля прогонов ({rates['n_runs']}), в которых внесенное значение найдено")
    print(f"{'значение':>9} {'три сигмы':>10} {'Тьюки':>7}")
    for v, a, b in zip([5, -4, 3.3, 2.99, -3], rates["sigma"], rates["tukey"]):
        print(f"{v:>+9.2f} {a:>10.3f} {b:>7.3f}")
    print(f"ложных тревог на 195 регулярных значений в среднем за прогон: "
          f"три сигмы {rates['false_sigma']:.3f}, Тьюки {rates['false_tukey']:.3f}")
    print()


# Вывод результатов задания 1
def print_task_1(rule: dict) -> None:
    print(f"выборочное среднее = {rule['mean']:+.4f}, s = {rule['std']:.4f}, порог 3s = {3 * rule['std']:.4f}")
    print(f"{'№ ст.':>6} {'значение':>9} {'|x-mean|/s':>11} {'аномальное':>11}")
    for r in rule["rows"]:
        print(f"{r['number']:>6} {r['value']:>+9.3f} {r['z']:>11.3f} {'да' if r['anomaly'] else 'нет':>11}")
    print()


# Вывод результатов задания 2
def print_task_2(box: dict) -> None:
    print(f"LQ = {box['lq']:+.4f}, med = {box['med']:+.4f}, UQ = {box['uq']:+.4f}, IQR = {box['iqr']:.4f}")
    print(f"пороги отбраковки: x_L = {box['x_low']:+.4f}, x_U = {box['x_up']:+.4f}")
    print("аномальные по боксплоту Тьюки:    ", np.round(box["outliers"], 3))
    print("аномальные по правилу трех сигм:   ", np.round(box["sigma_outliers"], 3))
    print("найдены только боксплотом Тьюки:   ", np.round(box["only_tukey"], 3))
    print()


# Вспомогательная: графики задания 2
def draw_outliers(x: np.ndarray, rule: dict, box: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), dpi=110, gridspec_kw=dict(width_ratios=[1, 2]))

    ax1.boxplot(x, whis=1.5, flierprops=dict(marker="o", markerfacecolor="tab:red", markeredgecolor="tab:red"))
    for i, v in enumerate(box["outliers"]):
        ax1.annotate(f"{v:+.2f}", (1, v), xytext=(8 + 40 * (i % 2), 0), textcoords="offset points",
                     fontsize=9, va="center")
    ax1.set_title("Боксплот Тьюки", fontsize=13)
    ax1.set_ylabel("значение")
    ax1.set_xticks([])

    k = np.arange(1, len(x) + 1)
    is_sigma = np.abs(x - rule["mean"]) > 3 * rule["std"]
    is_tukey = (x < box["x_low"]) | (x > box["x_up"])
    ax2.plot(k[~is_tukey & ~is_sigma], x[~is_tukey & ~is_sigma], ".", color="tab:blue", label="регулярные")
    ax2.plot(k[is_tukey & ~is_sigma], x[is_tukey & ~is_sigma], "o", color="tab:orange", label="только Тьюки")
    ax2.plot(k[is_sigma], x[is_sigma], "o", color="tab:red", label="три сигмы (и Тьюки)")
    for level in (rule["mean"] - 3 * rule["std"], rule["mean"] + 3 * rule["std"]):
        ax2.axhline(level, color="tab:red", linestyle="--", linewidth=1)
    for level in (box["x_low"], box["x_up"]):
        ax2.axhline(level, color="tab:orange", linestyle=":", linewidth=1.5)
    ax2.plot([], [], "--", color="tab:red", label=r"$\bar{x} \pm 3s$")
    ax2.plot([], [], ":", color="tab:orange", label=r"$x_L$, $x_U$")
    ax2.set_title("Отсортированный ряд: правило трех сигм и пороги Тьюки", fontsize=13)
    ax2.set_xlabel("номер порядковой статистики")
    ax2.set_ylabel("значение")
    ax2.legend(loc="upper left", framealpha=0.9)

    fig.tight_layout()


def main() -> None:
    x = np.sort(generate_series(np.random.default_rng(42)))

    print("=" * 30, "ЗАДАНИЕ 1", "=" * 30)
    rule = task_1(x)
    print_task_1(rule)

    print("=" * 30, "ЗАДАНИЕ 2", "=" * 30)
    box = task_2(x, rule)
    print_task_2(box)
    print_rates(estimate_rates())

    draw_outliers(x, rule, box)
    plt.show()


if __name__ == "__main__":
    main()