import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


# Вспомогательная: МНК-оценки и характеристики множественной регрессии (лекции 19, 20)
def linear_regression(X: np.ndarray, y: np.ndarray) -> dict:
    n, k = X.shape
    p = k - 1
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    y_hat = X @ beta

    rss = float(np.sum((y - y_hat) ** 2))
    tss = float(np.sum((y - y.mean()) ** 2))
    s2 = rss / (n - p - 1)
    se = np.sqrt(s2 * np.diag(np.linalg.inv(X.T @ X)))
    t = beta / se

    f = ((tss - rss) / p) / (rss / (n - p - 1)) if p > 0 else float("nan")
    return dict(
        beta=beta, y_hat=y_hat, n=n, p=p,
        rss=rss, tss=tss, rse=float(np.sqrt(s2)), r2=1 - rss / tss,
        se=se, t=t, p_values=2 * stats.t.sf(np.abs(t), n - p - 1),
        f=f, f_p_value=float(stats.f.sf(f, p, n - p - 1)) if p > 0 else float("nan"),
    )


# Вспомогательная: загрузка среднегодовых температур
def load_temperature(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    years = np.array([int(r["year"]) for r in rows])
    temp = np.array([float(r["temperature"]) for r in rows])
    return years, temp


# Задание 1: модель y = 1 + 3x1 - 2x2 + x3 + e и множественная регрессия
def task_1(n: int = 200) -> tuple[np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(42)

    x = rng.normal(0, 1, (n, 3))
    e = rng.normal(0, 1, n)
    y = 1 + 3 * x[:, 0] - 2 * x[:, 1] + x[:, 2] + e

    X = np.column_stack([np.ones(n), x])
    return X, y, linear_regression(X, y)


# Задание 3: полиномиальная регрессия температуры и прямой отбор степени
def task_3(years: np.ndarray, temp: np.ndarray, max_degree: int = 6, alpha: float = 0.05) -> dict:
    z = (years - years.mean()) / years.std()

    models = []
    for degree in range(1, max_degree + 1):
        X = np.column_stack([z ** j for j in range(degree + 1)])
        models.append(linear_regression(X, temp))

    chosen = 0
    for degree in range(1, max_degree + 1):
        if models[degree - 1]["p_values"][-1] < alpha:
            chosen = degree
        else:
            break

    return dict(z=z, models=models, chosen=chosen, alpha=alpha)


# Вывод результатов заданий 1 и 2
def print_task_1_2(res: dict) -> None:
    true_beta = [1, 3, -2, 1]
    print(f"{'коэф.':<6} {'истинный':>9} {'оценка':>9} {'SE':>7} {'t':>8} {'p-value':>9}")
    for j, (b, se, t, pv) in enumerate(zip(res["beta"], res["se"], res["t"], res["p_values"])):
        print(f"beta{j:<2} {true_beta[j]:>9} {b:>+9.4f} {se:>7.4f} {t:>+8.2f} {pv:>9.2g}")
    print()
    print("=" * 30, "ЗАДАНИЕ 2", "=" * 30)
    n, p = res["n"], res["p"]
    print(f"n = {n}, p = {p}")
    print(f"RSS = {res['rss']:.2f}, TSS = {res['tss']:.2f}")
    print(f"RSE = sqrt(RSS / (n - p - 1)) = {res['rse']:.4f}")
    print(f"R^2 = (TSS - RSS) / TSS = {res['r2']:.4f}")
    print(f"F = {res['f']:.1f}, p-value = {res['f_p_value']:.2g} (H0: beta1 = beta2 = beta3 = 0)")
    print()


# Вывод результатов задания 3
def print_task_3(res: dict, years: np.ndarray) -> None:
    print(f"Тюмень, среднегодовая температура, {years[0]}-{years[-1]} ({len(years)} лет)")
    print(f"прямой отбор: добавляем степень, пока p-value нового коэффициента < {res['alpha']}")
    print(f"{'степень':>7} {'RSS':>8} {'RSE':>7} {'R^2':>7} {'p-value нового':>15}")
    for degree, m in enumerate(res["models"], start=1):
        print(f"{degree:>7} {m['rss']:>8.3f} {m['rse']:>7.4f} {m['r2']:>7.4f} {m['p_values'][-1]:>15.3f}")
    print(f"выбрана степень {res['chosen']}")


# Вспомогательная: графики задания 1
def draw_task_1(res: dict, y: np.ndarray) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=110)
    ax1.plot(res["y_hat"], y, ".", color="tab:blue", alpha=0.7)
    lim = [min(y.min(), res["y_hat"].min()), max(y.max(), res["y_hat"].max())]
    ax1.plot(lim, lim, color="tab:red", label=r"$y = \hat y$")
    ax1.set_xlabel(r"прогноз $\hat y$")
    ax1.set_ylabel("наблюдаемое y")
    ax1.set_title("Задание 1. Наблюдаемые и предсказанные значения")
    ax1.legend()

    resid = y - res["y_hat"]
    ax2.plot(res["y_hat"], resid, ".", color="tab:blue", alpha=0.7)
    ax2.axhline(0, color="black", linewidth=1)
    ax2.set_xlabel(r"прогноз $\hat y$")
    ax2.set_ylabel(r"остаток $y - \hat y$")
    ax2.set_title("Задание 2. Остатки")
    fig.tight_layout()


# Вспомогательная: графики задания 3
def draw_task_3(res: dict, years: np.ndarray, temp: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=110)
    ax.plot(years, temp, "o", color="tab:gray", markersize=4, label="среднегодовая температура")
    colors = ["tab:blue", "tab:green", "tab:orange", "tab:purple"]
    for degree, color in zip(range(1, 5), colors):
        m = res["models"][degree - 1]
        style = dict(linewidth=2.5) if degree == res["chosen"] else dict(linewidth=1.2, linestyle="--")
        ax.plot(years, m["y_hat"], color=color, label=f"степень {degree}, $R^2$ = {m['r2']:.3f}", **style)
    ax.set_xlabel("год")
    ax.set_ylabel("температура, °C")
    ax.set_title(f"Задание 3. Полиномиальная регрессия (выбрана степень {res['chosen']})")
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()


def main() -> None:
    data = Path(__file__).resolve().parent.parent / "data" / "tyumen_annual_temperature.csv"

    print("=" * 30, "ЗАДАНИЕ 1", "=" * 30)
    X, y, res_1 = task_1()
    print_task_1_2(res_1)
    draw_task_1(res_1, y)

    print("=" * 30, "ЗАДАНИЕ 3", "=" * 30)
    years, temp = load_temperature(data)
    years, temp = years[-40:], temp[-40:]
    res_3 = task_3(years, temp)
    print_task_3(res_3, years)
    draw_task_3(res_3, years, temp)

    plt.show()


if __name__ == "__main__":
    main()
