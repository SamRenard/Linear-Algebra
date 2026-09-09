"""
Day 34 — Validation against NumPy (np.linalg)
NIZAM AI · 150-Day AI Engineering Protocol

Cross-checks the from-scratch `rank` and `null_space_dimension`
implementations against np.linalg.matrix_rank, and validates
solve_linear_system's classification against np.linalg.solve /
np.linalg.lstsq behavior.
"""

import numpy as np
from rank_null_space import rank, null_space_dimension, is_invertible, solve_linear_system


def to_numpy(m):
    return np.array(m, dtype=float)


def run_rank_case(m, label: str) -> bool:
    expected = np.linalg.matrix_rank(to_numpy(m))
    actual = rank(m)
    ok = expected == actual
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] rank — {label}  expected={expected} actual={actual}")
    return ok


def run_nullspace_case(m, label: str) -> bool:
    n_cols = len(m[0])
    expected = n_cols - np.linalg.matrix_rank(to_numpy(m))
    actual = null_space_dimension(m)
    ok = expected == actual
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] null_space_dim — {label}  expected={expected} actual={actual}")
    return ok


def random_matrix(rows: int, cols: int):
    return np.random.uniform(-10, 10, size=(rows, cols)).tolist()


def main() -> None:
    np.random.seed(3)
    results = []

    # Full rank square matrix
    A = [[1, 2, 3], [0, 1, 4], [5, 6, 0]]
    results.append(run_rank_case(A, "full rank 3x3"))
    results.append(run_nullspace_case(A, "full rank 3x3"))

    # Rank-deficient matrix (row3 = row1 + row2)
    B = [[1, 2, 3], [4, 5, 6], [5, 7, 9]]
    results.append(run_rank_case(B, "rank-deficient 3x3"))
    results.append(run_nullspace_case(B, "rank-deficient 3x3"))

    # Rectangular matrix
    C = [[1, 2, 3, 4], [2, 4, 6, 8]]  # rank 1, row2 = 2*row1
    results.append(run_rank_case(C, "rectangular 2x4 (dependent rows)"))
    results.append(run_nullspace_case(C, "rectangular 2x4"))

    # Random full-rank matrices
    for n in [2, 3, 4]:
        m = random_matrix(n, n)
        results.append(run_rank_case(m, f"random {n}x{n}"))

    # is_invertible sanity checks
    results.append(is_invertible(A) is True)
    print(f"[{'PASS' if results[-1] else 'FAIL'}] is_invertible(A) == True")
    results.append(is_invertible(B) is False)
    print(f"[{'PASS' if results[-1] else 'FAIL'}] is_invertible(B) == False")

    # Linear system classification
    unique_check = solve_linear_system(A, [1, 2, 3]) == "Unique solution (full rank)"
    results.append(unique_check)
    print(f"[{'PASS' if unique_check else 'FAIL'}] full-rank system -> unique solution")

    # Inconsistent system: B is rank-deficient, pick a b outside its column space
    inconsistent = solve_linear_system(B, [1, 0, 0])
    inconsistent_ok = "No solution" in inconsistent
    results.append(inconsistent_ok)
    print(f"[{'PASS' if inconsistent_ok else 'FAIL'}] rank-deficient system with incompatible b -> no solution")

    print(f"\n{sum(results)}/{len(results)} checks passed.")
    if not all(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
