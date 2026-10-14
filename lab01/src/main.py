def task_1() -> tuple[list[int], list[int]]:
    x = list(range(-10, 6))
    y = list(range(-5, 11))

    return x, y


def task_2(x: list[int], y: list[int]) -> tuple[list[int], list[int]]:
    z = []
    z_sorted = []

    for xi, yi in zip(x, y):
        z.extend([xi, yi]) 
    z_sorted = z.copy()
    z_sorted.sort();
    return z, z_sorted


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
        raise ValueError(f"unknown norm kind: {kind}")



def task_3(x: list[int], y: list[int], z: list[int]) -> dict[str, dict[str, float]]:
    result = {}
    vectors = {"x": x, "y": y, "z": z}
    for name, vec in vectors.items():
        result[name] = {}
        for n in ["l1", "l2", "inf"]:
            result[name][n] = norm(vec, n)
    return result



def factorial(n: int) -> int:
    if n < 0:
        raise ValueError(f"{n} is negative")
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result


def task_4(n: int) -> list[int]:
    result = []
    for i in range(n + 1):
        result.append(factorial(i))
    return result


def task_5() -> tuple[int, int, float, int]:
    print("Введите элементы через пробел.")
    vector = list(map(int, input().split()))

    print("Введите веса через пробел (в таком же количестве сколько и элементов выше).")
    weights = list(map(float, input().split()))

    if len(vector) != len(weights):
        raise ValueError("The number of elements of the weights and the vector does not match")

    return min(vector), max(vector), norm(vector, "w1", weights), sum(vector)


def main():
    print("="*30, "TASK 1", "="*30)
    x, y = task_1()
    print(f"x = {x}")
    print(f"y = {y}")

    print("="*30, "TASK 2", "="*30)
    z, z_sorted = task_2(x, y)
    print(f"z = {z}\nAfter sort\nz = {z_sorted}")

    print("="*30, "TASK 3", "="*30)    
    for name, norms in task_3(x, y, z).items():
        for kind, value in norms.items():
            print(f"{name} ({kind}) = {value}")
        print()
        
    print("="*30, "TASK 4", "="*30)
    result = task_4(11)
    for i, n in enumerate(result):
        print(f"{i}! = {n}")

    print("="*30, "TASK 5", "="*30)
    vec_min, vec_max, vec_norm, vec_sum = task_5()
    print(f"MIN: {vec_min}\nMAX: {vec_max}\nW_NORM: {vec_norm}\nSUM: {vec_sum}")


if __name__ == "__main__":
    main()