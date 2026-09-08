"""
Day 33 — Validation against NumPy (np.linalg)
NIZAM AI · 150-Day AI Engineering Protocol

Cross-checks the from-scratch `determinant` and `inverse` implementations
against NumPy's np.linalg.det and np.linalg.inv using fixed and random
test cases.
"""

import numpy as np
from determinant_inverse import determinant, inverse


def to_numpy(m):
    return np.array(m, dtype=float)


def run_det_case(m, label: str) -> bool:
    expected = np.linalg.det(to_numpy(m))
    actual = determinant(m)
    ok = np.isclose(expected, actual)
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] det — {label}  expected={expected:.4f} actual={actual:.4f}")
    return bool(ok)


def run_inverse_case(m, label: str) -> bool:
    expected = np.linalg.inv(to_numpy(m))
    actual = to_numpy(inverse(m))
    ok = np.allclose(expected, actual)
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] inverse — {label}")
    if not ok:
        print("  expected:\n", expected)
        print("  actual:\n", actual)
    return ok


def random_invertible_matrix(n: int):
    """Generate a random n x n matrix, retrying if it happens to be singular."""
    while True:
        m = np.random.uniform(-10, 10, size=(n, n))
        if abs(np.linalg.det(m)) > 1e-6:
            return m.tolist()


def main() -> None:
    np.random.seed(7)
    results = []

    # Fixed 2x2 cases
    results.append(run_det_case([[4, 7], [2, 6]], "fixed 2x2"))
    results.append(run_inverse_case([[4, 7], [2, 6]], "fixed 2x2"))

    # Fixed 3x3 case
    B = [[1, 2, 3], [0, 1, 4], [5, 6, 0]]
    results.append(run_det_case(B, "fixed 3x3"))
    results.append(run_inverse_case(B, "fixed 3x3"))

    # Random invertible matrices
    for n in [2, 2, 3, 3]:
        m = random_invertible_matrix(n)
        results.append(run_det_case(m, f"random {n}x{n}"))
        results.append(run_inverse_case(m, f"random {n}x{n}"))

    # Singular matrix should raise on inverse
    singular = [[1, 2], [2, 4]]
    try:
        inverse(singular)
        print("[FAIL] singular matrix did not raise ValueError")
        results.append(False)
    except ValueError:
        print("[PASS] singular matrix correctly raised ValueError on inverse")
        results.append(True)

    # A @ A^-1 should equal identity
    A = [[4, 7], [2, 6]]
    A_inv = inverse(A)
    identity_check = to_numpy(A) @ to_numpy(A_inv)
    ok = np.allclose(identity_check, np.eye(2))
    print(f"[{'PASS' if ok else 'FAIL'}] A @ A^-1 == Identity")
    results.append(ok)

    print(f"\n{sum(results)}/{len(results)} checks passed.")
    if not all(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
