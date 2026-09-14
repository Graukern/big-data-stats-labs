def task_1() -> tuple[list[int], list[int]]:
    x = list(range(-10, 6))
    y = list(range(-5, 11))

    return x, y


def task_2(x: list[int], y: list[int]) -> list[int]:
    z = []
    for xi, yi in zip(x, y):
        z.extend([xi, yi]) 
    z.sort()
    return z


def norm(vec: list[int], kind: str, weights: list[float] | None = None) -> float:
    if kind == 'l1':
        return sum(abs(x) for x in vec)
    elif kind == "l2":
        return sum(x * x for x in vec) ** 0.5
    elif kind == "inf":
        return max(abs(x) for x in vec)
    elif kind == "w1":
        if weights is None:
            raise ValueError("weights must be provided for w1 norm")
        else:
            return sum(w * abs(x) for w, x in zip(weights, vec))
    else:
        raise ValueError (f"unknown norm kind: {kind}")


def main():
    print("="*30, "TASK 1", "="*30)
    x, y = task_1()
    print(f"x = {x}")
    print(f"y = {y}")

    print("="*30, "TASK 2", "="*30)
    z = task_2(x, y)
    print(f"z = {z}")

    print("="*30, "TASK 3", "="*30)
    vectors = {"x": x, "y": y, "z": z}
    for name, vec in vectors.items():
        for n in ["l1", "l2", "inf"]:
            print(f"{name} ({n}) = {norm(vec, n)}")
        print()

if __name__ == "__main__":
    main()