"""
Day 37 — Automated correctness checks for the Week 5 review exercises.
NIZAM AI · 150-Day AI Engineering Protocol

Runs assertion-based checks on the core claims made in
week5_review_exercises.py, so "I understand this" is backed by a
passing test suite, not just eyeballed output.
"""

import numpy as np


def test_matmul_noncommutative():
    A = np.array([[2, 0], [0, 3]])
    B = np.array([[0, -1], [1, 0]])
    assert not np.array_equal(A @ B, B @ A)
    print("[PASS] matrix multiplication is non-commutative")


def test_determinant_matches_area_scale():
    A = np.array([[3, 1], [0, 2]])
    assert np.isclose(np.linalg.det(A), 6.0)
    print("[PASS] determinant correctly computed")


def test_inverse_solves_system():
    A = np.array([[2, 1], [1, 3]], dtype=float)
    b = np.array([5, 10], dtype=float)
    x = np.linalg.inv(A) @ b
    assert np.allclose(A @ x, b)
    print("[PASS] A^-1 @ b solves Ax = b")


def test_singular_matrix_detected():
    S = np.array([[2, 4], [1, 2]])
    assert np.isclose(np.linalg.det(S), 0.0)
    assert np.linalg.matrix_rank(S) < S.shape[0]
    print("[PASS] singular matrix correctly has det=0 and rank<n")


def test_rank_and_nullity():
    M = np.array([[1, 2, 3], [2, 4, 6], [1, 0, 1]])
    rank = np.linalg.matrix_rank(M)
    null_dim = M.shape[1] - rank
    assert rank == 2
    assert null_dim == 1
    print("[PASS] rank-nullity theorem holds")


def test_system_classification():
    A = np.array([[1, 2], [2, 4]])
    rank_a = np.linalg.matrix_rank(A)

    b_consistent = np.array([1, 2])
    aug_consistent = np.column_stack([A, b_consistent])
    assert np.linalg.matrix_rank(aug_consistent) == rank_a  # infinite solutions

    b_inconsistent = np.array([1, 0])
    aug_inconsistent = np.column_stack([A, b_inconsistent])
    assert np.linalg.matrix_rank(aug_inconsistent) > rank_a  # no solution
    print("[PASS] Ax=b classification (infinite vs no solution) correct")


def test_eigenpairs_satisfy_definition():
    A = np.array([[4, 1], [2, 3]], dtype=float)
    eigvals, eigvecs = np.linalg.eig(A)
    for i in range(len(eigvals)):
        v = eigvecs[:, i]
        assert np.allclose(A @ v, eigvals[i] * v, atol=1e-6)
    print("[PASS] all eigenpairs satisfy A@v = lambda*v")


def test_eigenvalue_stability_link():
    A_stable = np.array([[0.5, 0], [0, 0.8]])
    A_unstable = np.array([[1.2, 0], [0, 0.9]])
    v = np.array([1.0, 1.0])

    v_stable, v_unstable = v.copy(), v.copy()
    for _ in range(20):
        v_stable = A_stable @ v_stable
        v_unstable = A_unstable @ v_unstable

    assert np.linalg.norm(v_stable) < 1.0       # shrinks toward zero
    assert np.linalg.norm(v_unstable) > 10.0     # grows unbounded
    print("[PASS] |eigenvalue|<1 shrinks, |eigenvalue|>1 grows (stability link)")


def test_svd_reconstruction_improves_with_k():
    rng = np.random.default_rng(0)
    M = rng.normal(size=(6, 4))
    U, S, Vt = np.linalg.svd(M, full_matrices=False)

    errors = []
    for k in [1, 2, 4]:
        approx = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
        errors.append(np.linalg.norm(M - approx))

    assert errors[0] >= errors[1] >= errors[2]
    assert np.isclose(errors[2], 0, atol=1e-6)  # full rank = exact
    print("[PASS] SVD reconstruction error shrinks monotonically with k")


def test_svd_pca_equivalence():
    rng = np.random.default_rng(1)
    data = rng.normal(size=(50, 3)) @ rng.normal(size=(3, 3))
    centered = data - data.mean(axis=0)

    _, S, _ = np.linalg.svd(centered, full_matrices=False)
    svd_variance = (S ** 2) / (data.shape[0] - 1)

    cov = np.cov(centered, rowvar=False)
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]

    assert np.allclose(svd_variance, eigvals, atol=1e-6)
    print("[PASS] SVD singular values^2/(n-1) == covariance matrix eigenvalues")


if __name__ == "__main__":
    tests = [
        test_matmul_noncommutative,
        test_determinant_matches_area_scale,
        test_inverse_solves_system,
        test_singular_matrix_detected,
        test_rank_and_nullity,
        test_system_classification,
        test_eigenpairs_satisfy_definition,
        test_eigenvalue_stability_link,
        test_svd_reconstruction_improves_with_k,
        test_svd_pca_equivalence,
    ]
    passed = 0
    for t in tests:
        t()
        passed += 1
    print(f"\n{passed}/{len(tests)} checks passed.")
