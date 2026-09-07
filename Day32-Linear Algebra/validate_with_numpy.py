"""
Day 32 — Validation against NumPy
NIZAM AI · 150-Day AI Engineering Protocol

Cross-checks the from-scratch `matmul` implementation against
NumPy's optimized matrix multiplication (@ operator) using
random test cases.
"""

import numpy as np
from matrix_multiplication import matmul


def to_numpy(m):
    return np.array(m, dtype=float)


def run_case(a, b, label: str) -> bool:
    expected = to_numpy(a) @ to_numpy(b)
    actual = to_numpy(matmul(a, b))
    ok = np.allclose(expected, actual)
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {label}  shape={actual.shape}")
    if not ok:
        print("  expected:\n", expected)
        print("  actual:\n", actual)
    return ok


def random_matrix(rows: int, cols: int):
    return np.random.uniform(-10, 10, size=(rows, cols)).tolist()


def main() -> None:
    np.random.seed(42)
    results = []

    # Fixed known case
    results.append(run_case([[1, 2, 3], [4, 5, 6]], [[7, 8], [9, 10], [11, 12]], "fixed 2x3 . 3x2"))

    # Square matrices
    results.append(run_case([[1, 2], [3, 4]], [[0, 1], [1, 0]], "square 2x2"))

    # Random shapes
    for m, n, p in [(3, 4, 2), (5, 5, 5), (1, 6, 3), (7, 2, 4)]:
        a, b = random_matrix(m, n), random_matrix(n, p)
        results.append(run_case(a, b, f"random {m}x{n} . {n}x{p}"))

    # Dimension mismatch should raise
    try:
        matmul([[1, 2]], [[1, 2]])
        print("[FAIL] dimension mismatch did not raise ValueError")
        results.append(False)
    except ValueError:
        print("[PASS] dimension mismatch correctly raised ValueError")
        results.append(True)

    print(f"\n{sum(results)}/{len(results)} checks passed.")
    if not all(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
