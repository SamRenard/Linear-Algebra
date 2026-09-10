"""
Day 35 — Validation against NumPy (np.linalg.eig)
NIZAM AI · 150-Day AI Engineering Protocol

Cross-checks the from-scratch `eigenvalues_2x2` and eigenvector
computation against np.linalg.eig using fixed and random test cases.
"""

import numpy as np
from eigen import eigenvalues_2x2, eigen_decompose_2x2, verify_eigenpair


def to_numpy(m):
    return np.array(m, dtype=float)


def run_eigenvalue_case(m, label: str) -> bool:
    expected = sorted(np.linalg.eigvals(to_numpy(m)), key=lambda x: x.real)
    actual = sorted(eigenvalues_2x2(m), key=lambda x: x.real)
    ok = all(
        abs(e.real - a.real) < 1e-6 and abs(e.imag - a.imag) < 1e-6
        for e, a in zip(expected, actual)
    )
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] eigenvalues — {label}  expected={expected} actual={actual}")
    return ok


def run_eigenpair_verification(m, label: str) -> bool:
    """Confirm each from-scratch (eigenvalue, eigenvector) pair satisfies A@v = lambda*v."""
    pairs = eigen_decompose_2x2(m)
    all_ok = all(verify_eigenpair(m, lam, v) for lam, v in pairs)
    status = "PASS" if all_ok else "FAIL"
    print(f"[{status}] eigenpair verification — {label}")
    return all_ok


def run_eigenvector_direction_case(m, label: str) -> bool:
    """
    Eigenvectors are only defined up to scale/sign, so we compare
    NumPy's and our eigenvectors by checking they point along the
    same line (cross product ~ 0), not by exact value.
    """
    np_vals, np_vecs = np.linalg.eig(to_numpy(m))
    pairs = eigen_decompose_2x2(m)

    all_ok = True
    for lam, v in pairs:
        # find the matching numpy eigenvalue
        idx = np.argmin(np.abs(np_vals.real - lam))
        np_v = np_vecs[:, idx].real
        cross = v[0] * np_v[1] - v[1] * np_v[0]
        ok = abs(cross) < 1e-6
        all_ok = all_ok and ok

    status = "PASS" if all_ok else "FAIL"
    print(f"[{status}] eigenvector direction match — {label}")
    return all_ok


def random_symmetric_matrix():
    """Symmetric matrices always have real eigenvalues — safer for this comparison."""
    a = np.random.uniform(-10, 10)
    b = np.random.uniform(-10, 10)
    d = np.random.uniform(-10, 10)
    return [[a, b], [b, d]]


def main() -> None:
    np.random.seed(11)
    results = []

    A = [[4, 1], [2, 3]]
    results.append(run_eigenvalue_case(A, "A (asymmetric, real eigenvalues)"))
    results.append(run_eigenpair_verification(A, "A"))
    results.append(run_eigenvector_direction_case(A, "A"))

    B = [[5, 0], [0, 2]]
    results.append(run_eigenvalue_case(B, "B (diagonal)"))
    results.append(run_eigenpair_verification(B, "B"))

    # Symmetric matrix (like a covariance matrix in PCA)
    C = [[6, 2], [2, 3]]
    results.append(run_eigenvalue_case(C, "C (symmetric)"))
    results.append(run_eigenpair_verification(C, "C"))
    results.append(run_eigenvector_direction_case(C, "C"))

    # Random symmetric matrices
    for i in range(3):
        m = random_symmetric_matrix()
        results.append(run_eigenvalue_case(m, f"random symmetric #{i+1}"))
        results.append(run_eigenpair_verification(m, f"random symmetric #{i+1}"))

    print(f"\n{sum(results)}/{len(results)} checks passed.")
    if not all(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
