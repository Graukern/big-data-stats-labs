import csv
import json
import urllib.parse
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Вспомогательная: метод Прони, этапы 1-3 из лекции (для задания 1)
def prony_method(
        x: np.ndarray,
        m: int,
        dt: float,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(x)

    X = np.column_stack([x[m - k : n - k] for k in range(1, m + 1)])
    a = np.linalg.lstsq(X, -x[m:], rcond=None)[0]

    z = np.roots(np.r_[1, a])
    lambdas = np.log(np.abs(z)) / dt
    freqs = np.arctan2(z.imag, z.real) / (2 * np.pi * dt)

    Z = z[None, :] ** np.arange(1, n + 1)[:, None]
    h = np.linalg.lstsq(Z, x.astype(complex), rcond=None)[0]

    return lambdas, freqs, np.abs(h), np.angle(h)


# Вспомогательная: восстановление ряда по параметрам Прони (для задания 1)
def prony_restore(
        lambdas: np.ndarray,
        freqs: np.ndarray,
        amplitudes: np.ndarray,
        phases: np.ndarray,
        dt: float,
        n: int,
        ) -> np.ndarray:
    t = dt * np.arange(1, n + 1)
    restored = np.zeros(n, dtype=complex)
    for lam, f, a, p in zip(lambdas, freqs, amplitudes, phases):
        restored += a * np.exp(lam * t + 1j * (2 * np.pi * f * t + p))
    return restored


# Вспомогательная: загрузка среднесуточных температур с кэшем в csv (для задания 2)
def load_weather(
        lat: float,
        lon: float,
        timezone: str,
        start_date: str,
        end_date: str,
        cache: Path,
        ) -> tuple[np.ndarray, np.ndarray]:
    if not cache.exists():
        query = urllib.parse.urlencode({
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "daily": "temperature_2m_mean",
            "timezone": timezone,
        })
        with urllib.request.urlopen(f"https://archive-api.open-meteo.com/v1/archive?{query}") as response:
            daily = json.load(response)["daily"]

        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["date", "temperature"])
            writer.writerows(zip(daily["time"], daily["temperature_2m_mean"]))

    with open(cache, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    dates = np.array([r["date"] for r in rows], dtype="datetime64[D]")
    temp = np.array([float(r["temperature"]) if r["temperature"] else np.nan for r in rows])
    if np.isnan(temp).any():
        raise ValueError("в данных есть пропуски, измените период или удалите кэш")
    return dates, temp


# Вспомогательная: скользящее среднее без краев (для задания 2)
def moving_mean(x: np.ndarray, width: int) -> np.ndarray:
    y = np.full(len(x), np.nan)
    m = width // 2
    y[m : len(x) - m] = np.convolve(x, np.ones(2 * m + 1) / (2 * m + 1), mode="valid")
    return y


# Вспомогательная: размах R(n)/s(n) для первых n точек (для показателя Херста)
def rescaled_range(x: np.ndarray, n_min: int = 10) -> tuple[np.ndarray, np.ndarray]:
    sizes = np.arange(n_min, len(x) + 1)
    rs = np.empty(len(sizes))
    for j, n in enumerate(sizes):
        part = x[:n]
        deviations = np.cumsum(part - part.mean())
        rs[j] = (deviations.max() - deviations.min()) / part.std()
    return sizes, rs


# Вспомогательная: показатель Херста, регрессия y = Hz + c (для задания 2)
def hurst_exponent(x: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
    sizes, rs = rescaled_range(x)
    h, c = np.polyfit(np.log(sizes), np.log(rs), 1)
    return float(h), float(c), sizes, rs


# Вспомогательная: число поворотных точек (для задания 2)
def turning_points(r: np.ndarray) -> tuple[int, float, float]:
    n = len(r)
    a, b, c = r[:-2], r[1:-1], r[2:]
    is_turn = ((a < b) & (b > c)) | ((a > b) & (b < c))
    p = int(is_turn.sum())
    e_p = 2 / 3 * (n - 2)
    s_p = np.sqrt((16 * n - 29) / 90)
    return p, e_p, s_p


# Вспомогательная: коэффициент Кендэла (для задания 2)
def kendall_tau(r: np.ndarray) -> tuple[float, float]:
    n = len(r)
    big_p = sum(int((r[i + 1 :] > r[i]).sum()) for i in range(n - 1))
    tau = 4 * big_p / (n * (n - 1)) - 1
    s_tau = np.sqrt(2 * (2 * n + 5) / (9 * n * (n - 1)))
    return tau, s_tau


# Задание 1: разложение модельного ряда методом Прони
def task_1(n: int = 200, h: float = 0.02, m_list: tuple[int, ...] = (2, 4, 6, 8)) -> dict:
    i = np.arange(1, n + 1)
    x = sum(k * np.exp(-h * i / k) * np.cos(4 * np.pi * k * h * i + np.pi / k) for k in (1, 2, 3))

    rows = []
    for m in m_list:
        lambdas, freqs, amplitudes, phases = prony_method(x, m, h)
        restored = prony_restore(lambdas, freqs, amplitudes, phases, h, n)
        order = np.argsort(freqs)
        rows.append(dict(
            m=m,
            lambdas=lambdas[order], freqs=freqs[order],
            amplitudes=amplitudes[order], phases=phases[order],
            restored=restored,
            error=float(np.abs(x - restored).max()),
        ))
    return dict(t=h * i, x=x, rows=rows)


# Задание 2: анализ температуры на наличие тренда и сезонных колебаний
def task_2(dates: np.ndarray, temp: np.ndarray, widths: tuple[int, ...] = (31, 91, 365)) -> dict:
    n = len(temp)
    k = np.arange(n)

    slope, intercept = np.polyfit(k, temp, 1)
    linear = slope * k + intercept
    slope_se = np.std(temp - linear, ddof=2) / np.sqrt(np.sum((k - k.mean()) ** 2))
    smooth = {w: moving_mean(temp, w) for w in widths}

    detrended = temp - linear
    spectrum = np.fft.rfft(detrended)
    amplitude = np.abs(spectrum) * 2 / n
    freq = np.fft.rfftfreq(n, 1)
    j = int(np.argmax(amplitude[1:])) + 1
    period = 1 / freq[j]
    phase = float(np.angle(spectrum[j]))
    seasonal = amplitude[j] * np.cos(2 * np.pi * j * k / n + phase)
    day_max = int(round((-phase * period / (2 * np.pi)) % period))

    residual = detrended - seasonal
    series = {"исходный ряд": temp, "без тренда": detrended, "остатки": residual}

    stats = []
    for name, s in series.items():
        p, e_p, s_p = turning_points(s)
        tau, s_tau = kendall_tau(s)
        h, c, sizes, rs = hurst_exponent(s)
        stats.append(dict(
            name=name, mean=float(s.mean()), std=float(s.std(ddof=1)),
            p=p, e_p=e_p, s_p=s_p, tau=tau, s_tau=s_tau,
            random=abs(p - e_p) < 1.96 * s_p and abs(tau) < 1.96 * s_tau,
            hurst=h, hurst_c=c, sizes=sizes, rs=rs,
        ))

    return dict(
        dates=dates, temp=temp, linear=linear, slope_year=float(slope * 365.25),
        slope_year_se=float(slope_se * 365.25), smooth=smooth,
        freq=freq, amplitude=amplitude, j_main=j, period=float(period),
        seasonal_amp=float(amplitude[j]), seasonal=seasonal,
        date_max=dates[0] + np.timedelta64(day_max, "D"),
        detrended=detrended, residual=residual, stats=stats,
    )


# Вывод результатов задания 1
def print_task_1(res: dict) -> None:
    print("точные параметры модели (6 комплексных экспонент, пары k = 1, 2, 3):")
    print("  lambda = -1/k = -1, -0.5, -0.333;  f = +-2k = +-2, +-4, +-6;  A = k/2 = 0.5, 1, 1.5;  phi = +-pi/k")
    print()
    for r in res["rows"]:
        print(f"m = {r['m']}: max|x - восстановленный| = {r['error']:.2e}")
        for lam, f, a, p in zip(r["lambdas"], r["freqs"], r["amplitudes"], r["phases"]):
            print(f"  lambda = {lam:9.4f}, f = {f:9.4f}, A = {a:.4f}, phi = {p:8.4f}")
        print()


# Вывод результатов задания 2
def print_task_2(res: dict, city: str) -> None:
    n = len(res["temp"])
    print(f"{city}: {res['dates'][0]} - {res['dates'][-1]}, {n} дней")
    print()
    print(f"линейный тренд: {res['slope_year']:+.2f} +- {res['slope_year_se']:.2f} градусов в год (1 s.e.)")
    for w, y in res["smooth"].items():
        valid = y[~np.isnan(y)]
        print(f"скользящее среднее w={w:>3}: от {valid[0]:+6.2f} до {valid[-1]:+6.2f} градусов "
              f"(размах {valid.max() - valid.min():.2f})")
    print()
    print(f"главный период спектра Фурье: j = {res['j_main']}, T = {res['period']:.1f} дней, "
          f"амплитуда = {res['seasonal_amp']:.2f} градусов")
    print(f"максимум сезонной волны приходится на {res['date_max']}")
    print()
    print("случайность: |p - E(p)| < 1.96 s(p) и |tau| < 1.96 s(tau)")
    print(f"{'ряд':<14} {'среднее':>8} {'СКО':>7} {'p':>4} {'E(p)':>7} {'tau':>7} {'случаен':>8} {'H':>7}")
    for s in res["stats"]:
        print(f"{s['name']:<14} {s['mean']:>+8.3f} {s['std']:>7.3f} {s['p']:>4} {s['e_p']:>7.1f} "
              f"{s['tau']:>+7.3f} {'да' if s['random'] else 'НЕТ':>8} {s['hurst']:>7.3f}")


# Вспомогательная: графики задания 1
def draw_prony(res: dict) -> None:
    rows = res["rows"]
    fig, axes = plt.subplots(2, len(rows), figsize=(4.5 * len(rows), 7.5), dpi=100)
    for col, r in enumerate(rows):
        ax = axes[0, col]
        ax.plot(res["t"], res["x"], ".", color="tab:orange", label="модель")
        ax.plot(res["t"], r["restored"].real, color="tab:blue", label="Прони")
        ax.set_title(f"m = {r['m']}, ошибка {r['error']:.1e}", fontsize=11)
        ax.set_xlabel("t")
        ax.legend(loc="upper right")

        ax = axes[1, col]
        ax.stem(r["freqs"], r["amplitudes"])
        ax.set_title(f"m = {r['m']}: компоненты Прони", fontsize=11)
        ax.set_xlabel("частота f")
        ax.set_ylabel("A")
    fig.suptitle("Задание 1. Метод Прони", fontsize=13)
    fig.tight_layout()


# Вспомогательная: графики задания 2
def draw_temperature(res: dict, city: str) -> None:
    dates, temp = res["dates"], res["temp"]

    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=110)
    ax.plot(dates, temp, ".", markersize=3, color="tab:gray", label="температура")
    ax.plot(dates, res["linear"], color="black", linewidth=2, linestyle="--",
            label=f"линейный тренд ({res['slope_year']:+.2f} град/год)")
    for (w, y), color in zip(res["smooth"].items(), ["tab:green", "tab:blue", "tab:red"]):
        ax.plot(dates, y, color=color, linewidth=1.8, label=f"скользящее среднее w={w}")
    ax.set_title(f"Задание 2. {city}: среднесуточная температура и тренд", fontsize=13, pad=12)
    ax.set_ylabel("температура, °C")
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=110)
    ax1.plot(res["freq"][1:], res["amplitude"][1:], color="tab:blue")
    ax1.set_title("Амплитудный спектр Фурье (без тренда)", fontsize=12)
    ax1.set_xlabel("частота, 1/день")
    ax1.set_ylabel("амплитуда, °C")

    low = res["freq"] <= 0.05
    ax2.stem(res["freq"][low][1:], res["amplitude"][low][1:])
    jm = res["j_main"]
    ax2.plot(res["freq"][jm], res["amplitude"][jm], "o", color="tab:red", markersize=9,
             label=f"главный период {res['period']:.0f} дней")
    ax2.set_title("Низкие частоты", fontsize=12)
    ax2.set_xlabel("частота, 1/день")
    ax2.legend(loc="upper right")
    fig.tight_layout()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), dpi=110, sharex=True)
    ax1.plot(dates, res["detrended"], ".", markersize=3, color="tab:gray", label="ряд без тренда")
    ax1.plot(dates, res["seasonal"], color="tab:red", linewidth=2,
             label=f"сезонная волна (T = {res['period']:.0f} дней, A = {res['seasonal_amp']:.1f})")
    ax1.set_ylabel("°C")
    ax1.legend(loc="best", framealpha=0.9)
    ax1.set_title("Сезонная составляющая и остатки", fontsize=13, pad=12)
    ax2.plot(dates, res["residual"], linewidth=0.8, color="tab:blue")
    ax2.axhline(0, color="black", linewidth=1)
    ax2.set_ylabel("остатки, °C")
    fig.tight_layout()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=110)
    for row, s in zip(axes, (res["stats"][0], res["stats"][2])):
        sizes, rs, h, c = s["sizes"], s["rs"], s["hurst"], s["hurst_c"]
        row[0].plot(sizes, rs, ".", color="tab:orange", label="R/s")
        row[0].plot(sizes, np.exp(c) * sizes ** h, color="tab:blue", label=f"{np.exp(c):.2f} n^{h:.3f}")
        row[0].set_title(f"{s['name']}: R/s(n)", fontsize=11)
        row[0].set_xlabel("n")
        row[0].legend(loc="upper left")
        row[1].plot(np.log(sizes), np.log(rs), ".", color="tab:orange", label="y = ln(R/s)")
        row[1].plot(np.log(sizes), h * np.log(sizes) + c, color="tab:blue", label=f"y = {h:.3f}z + {c:.3f}")
        row[1].set_title(f"{s['name']}: H = {h:.3f}", fontsize=11)
        row[1].set_xlabel("z = ln(n)")
        row[1].legend(loc="upper left")
    fig.suptitle("Показатель Херста", fontsize=13)
    fig.tight_layout()


def main() -> None:
    city, lat, lon, timezone = "Санкт-Петербург", 59.9386, 30.3141, "Europe/Moscow"
    start_date, end_date = "2024-10-01", "2026-09-30"
    cache = Path(__file__).resolve().parent.parent / "data" / f"weather_{start_date}_{end_date}.csv"

    print("=" * 30, "ЗАДАНИЕ 1", "=" * 30)
    res_1 = task_1()
    print_task_1(res_1)
    draw_prony(res_1)

    print("=" * 30, "ЗАДАНИЕ 2", "=" * 30)
    dates, temp = load_weather(lat, lon, timezone, start_date, end_date, cache)
    res_2 = task_2(dates, temp)
    print_task_2(res_2, city)
    draw_temperature(res_2, city)

    plt.show()


if __name__ == "__main__":
    main()