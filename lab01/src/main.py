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


def main():
    print("="*30, "TASK 1", "="*30)
    x, y = task_1()
    print(f"x = {x}")
    print(f"y = {y}")

    print("="*30, "TASK 2", "="*30)
    z = task_2(x, y)
    print(f"z = {z}")


if __name__ == "__main__":
    main()