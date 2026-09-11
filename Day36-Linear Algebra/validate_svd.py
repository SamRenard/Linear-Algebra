"""
Day 36 — Validation: SVD Reconstruction Correctness & PCA Connection
NIZAM AI · 150-Day AI Engineering Protocol

1. Validates that our rank-k reconstruction matches manually computing
   U_k @ diag(S_k) @ Vt_k using NumPy's SVD directly (sanity check that
   compress_with_svd is doing exactly what it claims).
2. Confirms full-rank reconstruction (k = min(m, n)) recovers the
   original image almost exactly.
3. Demonstrates the PCA connection: PCA on a dataset is just SVD on the
   (mean-centered) data matrix, and the singular values relate directly
   to explained variance.
"""

import numpy as np
from svd_compression import compress_with_svd, make_synthetic_image


def run_case(image, k, label) -> bool:
    U, S, Vt = np.linalg.svd(image, full_matrices=False)
    expected = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    actual = compress_with_svd(image, k)
    ok = np.allclose(expected, actual)
    print(f"[{'PASS' if ok else 'FAIL'}] reconstruction matches manual SVD — {label}")
    return ok


def run_full_rank_case(image) -> bool:
    full_k = min(image.shape)
    reconstructed = compress_with_svd(image, full_k)
    ok = np.allclose(image, reconstructed, atol=1e-6)
    print(f"[{'PASS' if ok else 'FAIL'}] full-rank reconstruction recovers original")
    return ok


def run_error_decreases_with_k(image) -> bool:
    """Reconstruction error should monotonically decrease as k increases."""
    errors = []
    for k in [5, 20, 50, 100]:
        reconstructed = compress_with_svd(image, k)
        err = np.linalg.norm(image - reconstructed)
        errors.append(err)
    ok = all(errors[i] >= errors[i + 1] for i in range(len(errors) - 1))
    print(f"[{'PASS' if ok else 'FAIL'}] reconstruction error decreases as k increases  {errors}")
    return ok


def run_pca_connection_demo() -> bool:
    """
    PCA is SVD applied to mean-centered data. The singular values from
    SVD relate to PCA's explained variance: variance_i = S[i]^2 / (n-1).
    We confirm this equivalence against sklearn-style manual PCA.
    """
    np.random.seed(0)
    data = np.random.randn(100, 5) @ np.random.randn(5, 5)  # correlated features
    centered = data - data.mean(axis=0)

    # SVD-based "PCA"
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    svd_variance = (S ** 2) / (data.shape[0] - 1)

    # Direct covariance-eigenvalue PCA (Day 35 connection!)
    cov = np.cov(centered, rowvar=False)
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]

    ok = np.allclose(svd_variance, eigvals, atol=1e-6)
    print(f"[{'PASS' if ok else 'FAIL'}] SVD singular values <-> PCA eigenvalues of covariance match")
    return ok


def main() -> None:
    image = make_synthetic_image(size=80)  # smaller for fast tests
    results = []

    for k in [5, 20, 50]:
        results.append(run_case(image, k, f"k={k}"))

    results.append(run_full_rank_case(image))
    results.append(run_error_decreases_with_k(image))
    results.append(run_pca_connection_demo())

    print(f"\n{sum(results)}/{len(results)} checks passed.")
    if not all(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
