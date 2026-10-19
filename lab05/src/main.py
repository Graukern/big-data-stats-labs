import numpy as np


# Вспомогательная: выборки из распределений задания 2
def generate(name: str, rng: np.random.Generator, n_trials: int, n: int) -> np.ndarray:
    if name == "N(0,1)":
        return rng.normal(0, 1, (n_trials, n))
    if name == "C(0,1)":
        return rng.standard_cauchy((n_trials, n))
    mask = rng.random((n_trials, n)) < 0.1
    return np.where(mask, rng.standard_cauchy((n_trials, n)), rng.normal(0, 1, (n_trials, n)))


# Вспомогательная: оценка Хубера (k = 1.44), масштаб по MAD
def huber(x: np.ndarray, k: float = 1.44, n_iter: int = 100, tol: float = 1e-8) -> np.ndarray:
    t = np.median(x, axis=1)
    s = np.median(np.abs(x - t[:, None]), axis=1) / 0.6745
    for _ in range(n_iter):
        u = (x - t[:, None]) / s[:, None]
        w = np.minimum(1, k / np.maximum(np.abs(u), 1e-12))
        t_new = np.sum(w * x, axis=1) / np.sum(w, axis=1)
        if np.max(np.abs(t_new - t)) < tol:
            return t_new
        t = t_new
    return t


# Вспомогательная: двухэтапная оценка (боксплот Тьюки + среднее)
def two_step(x: np.ndarray) -> np.ndarray:
    lq, uq = np.percentile(x, [25, 75], axis=1)
    iqr = uq - lq
    x_low = np.maximum(x.min(axis=1), lq - 1.5 * iqr)
    x_up = np.minimum(x.max(axis=1), uq + 1.5 * iqr)
    keep = (x >= x_low[:, None]) & (x <= x_up[:, None])
    return np.sum(np.where(keep, x, 0), axis=1) / keep.sum(axis=1)


# Задание 3: среднее и дисперсия оценок по методу Монте-Карло
def task_3(n_trials: int = 10000, n: int = 100, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    estimators = {
        "среднее": lambda x: x.mean(axis=1),
        "медиана": lambda x: np.median(x, axis=1),
        "Хубер": huber,
        "двухэтапная": two_step,
    }
    result = {}
    for dist in ("N(0,1)", "C(0,1)", "0.9N+0.1C"):
        x = generate(dist, rng, n_trials, n)
        result[dist] = {}
        for name, func in estimators.items():
            theta = func(x)
            result[dist][name] = (float(np.mean(theta)), float(np.mean(theta ** 2) - np.mean(theta) ** 2))
    return result


def main() -> None:
    result = task_3()
    print("N = 10000 испытаний, n = 100, theta = 0")
    for dist, rows in result.items():
        print(f"\n{dist}")
        print(f"{'оценка':<12} {'E':>12} {'D':>14}")
        for name, (e, d) in rows.items():
            print(f"{name:<12} {e:>+12.5f} {d:>14.6g}")


if __name__ == "__main__":
    main()