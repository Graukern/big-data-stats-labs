import matplotlib.pyplot as plt
import numpy as np


# Задание 4: данные (возраст, лет; вес, кг)
def task_4_data() -> tuple[np.ndarray, np.ndarray]:
    age = np.array([3, 5, 7, 10, 12, 14, 16, 19, 22, 27, 33, 40, 46, 52, 58, 65], dtype=float)
    weight = np.array([15, 19, 23, 31, 38, 49, 55, 61, 66, 70, 78, 74, 83, 80, 86, 79], dtype=float)
    return age, weight


# Вспомогательная: простая линейная регрессия МНК
def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    b1 = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2)
    b0 = y.mean() - b1 * x.mean()
    return float(b0), float(b1)


# Задание 4: мультипликативная модель Y = alpha * X^gamma
def task_4(age: np.ndarray, weight: np.ndarray) -> dict:
    n = len(age)
    z, v = np.log(age), np.log(weight)
    b0, b1 = linear_fit(z, v)
    alpha, gamma = np.exp(b0), b1

    v_hat = b0 + b1 * z
    rss_log = np.sum((v - v_hat) ** 2)
    r2_log = 1 - rss_log / np.sum((v - v.mean()) ** 2)

    w_hat = alpha * age ** gamma
    residual = weight - w_hat
    rss = np.sum(residual ** 2)
    r2 = 1 - rss / np.sum((weight - weight.mean()) ** 2)
    rse = np.sqrt(rss / (n - 2))

    return dict(b0=b0, alpha=alpha, gamma=gamma, r2_log=r2_log, rse_log=np.sqrt(rss_log / (n - 2)),
                w_hat=w_hat, residual=residual, rss=rss, r2=r2, rse=rse)


def main() -> None:
    age, weight = task_4_data()
    res = task_4(age, weight)

    print(f"ln(alpha) = {res['b0']:.4f}, alpha = {res['alpha']:.4f}, gamma = {res['gamma']:.4f}")
    print(f"V = ln Y:  R^2 = {res['r2_log']:.4f}, RSE = {res['rse_log']:.4f}")
    print(f"Y:         R^2 = {res['r2']:.4f}, RSS = {res['rss']:.2f}, RSE = {res['rse']:.2f}")
    print(f"{'X':>4} {'Y':>4} {'Y^':>7} {'e':>7}")
    for x, y, yh, e in zip(age, weight, res["w_hat"], res["residual"]):
        print(f"{x:>4.0f} {y:>4.0f} {yh:>7.1f} {e:>+7.1f}")

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=110)
    t = np.linspace(age.min(), age.max(), 300)
    ax.plot(age, weight, "o", color="tab:blue", label="исходные данные")
    ax.plot(t, res["alpha"] * t ** res["gamma"], color="tab:red",
            label=rf"$\hat Y = {res['alpha']:.2f}\,X^{{{res['gamma']:.3f}}}$")
    ax.set_xlabel("возраст, лет")
    ax.set_ylabel("вес, кг")
    ax.set_title("Мультипликативная модель зависимости веса от возраста")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
