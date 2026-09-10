"""
Day 35 — Eigenvalues & Eigenvectors
NIZAM AI · 150-Day AI Engineering Protocol

Goal: compute eigenvalues (2x2 case) from scratch via the characteristic
equation, find corresponding eigenvectors, then validate against
NumPy's np.linalg.eig.

Core intuition: for most vectors, a matrix transformation rotates AND
scales them. Eigenvectors are the special directions that only get
scaled, never rotated off their own line: A @ v = lambda * v. The
eigenvalue lambda is that scaling factor. These directions are the
"natural axes" of the transformation — everything about how A behaves
becomes simple once you look along its eigenvectors.

This is why eigen-decomposition matters in practice: PCA finds the
directions of maximum variance in data via eigenvectors of the
covariance matrix; PageRank finds the dominant eigenvector of a
link-graph matrix; and neural network training stability is analyzed
via the eigenvalues of weight matrices (exploding/vanishing gradients).
"""

from __future__ import annotations
from typing import List, Tuple
import math

Matrix = List[List[float]]
Vector = List[float]


def eigenvalues_2x2(m: Matrix) -> List[complex]:
    """
    Solve the characteristic equation det(A - lambda*I) = 0 for a 2x2 matrix.

    For A = [[a, b], [c, d]]:
      det(A - lambda*I) = (a-lambda)(d-lambda) - bc = 0
      => lambda^2 - (a+d)*lambda + (ad - bc) = 0

    This is a quadratic in lambda: lambda^2 - trace*lambda + det = 0
    Solved via the quadratic formula. Eigenvalues can be complex
    (e.g. for pure rotation matrices), so we return complex numbers.
    """
    a, b = m[0]
    c, d = m[1]

    trace = a + d
    det = a * d - b * c

    discriminant = trace ** 2 - 4 * det

    if discriminant >= 0:
        sqrt_disc = math.sqrt(discriminant)
        lambda1 = (trace + sqrt_disc) / 2
        lambda2 = (trace - sqrt_disc) / 2
        return [complex(lambda1, 0), complex(lambda2, 0)]
    else:
        sqrt_disc = math.sqrt(-discriminant)
        real_part = trace / 2
        imag_part = sqrt_disc / 2
        return [complex(real_part, imag_part), complex(real_part, -imag_part)]


def eigenvector_for_real_eigenvalue(m: Matrix, lam: float) -> Vector:
    """
    Find an eigenvector for a given (real) eigenvalue by solving
    (A - lambda*I) v = 0, i.e. finding the null space direction.

    For 2x2: (a-lambda)*x + b*y = 0  =>  y = -(a-lambda)/b * x  (if b != 0)
    We normalize the resulting vector to unit length.
    """
    a, b = m[0]
    c, d = m[1]

    a_shifted = a - lam
    d_shifted = d - lam

    if abs(b) > 1e-9:
        x, y = 1.0, -a_shifted / b
    elif abs(c) > 1e-9:
        x, y = -d_shifted / c, 1.0
    else:
        # Diagonal matrix case: eigenvector is a standard basis vector
        x, y = (1.0, 0.0) if abs(a_shifted) < 1e-9 else (0.0, 1.0)

    norm = math.sqrt(x ** 2 + y ** 2)
    return [x / norm, y / norm]


def eigen_decompose_2x2(m: Matrix) -> List[Tuple[float, Vector]]:
    """
    Convenience wrapper: returns (eigenvalue, eigenvector) pairs for a
    2x2 matrix with real eigenvalues. Raises if eigenvalues are complex
    (i.e. the transformation includes a pure rotation with no real
    invariant direction).
    """
    eigvals = eigenvalues_2x2(m)
    results = []
    for lam in eigvals:
        if abs(lam.imag) > 1e-9:
            raise ValueError(
                f"Eigenvalue {lam} is complex — matrix has no real "
                f"eigenvector for this value (pure rotation component)."
            )
        vec = eigenvector_for_real_eigenvalue(m, lam.real)
        results.append((lam.real, vec))
    return results


def verify_eigenpair(m: Matrix, lam: float, v: Vector) -> bool:
    """Check that A @ v ≈ lambda * v (the defining eigenvector property)."""
    av = [
        m[0][0] * v[0] + m[0][1] * v[1],
        m[1][0] * v[0] + m[1][1] * v[1],
    ]
    lv = [lam * v[0], lam * v[1]]
    return all(abs(av[i] - lv[i]) < 1e-6 for i in range(2))


def print_matrix(m: Matrix, label: str = "") -> None:
    if label:
        print(f"{label}:")
    for row in m:
        print("  [" + ", ".join(f"{v:7.3f}" for v in row) + "]")
    print()


if __name__ == "__main__":
    # Symmetric matrix — always has real eigenvalues (common in PCA/covariance)
    A: Matrix = [
        [4, 1],
        [2, 3],
    ]
    print_matrix(A, "A")

    pairs = eigen_decompose_2x2(A)
    for lam, v in pairs:
        print(f"eigenvalue = {lam:.4f}   eigenvector = [{v[0]:.4f}, {v[1]:.4f}]")
        print(f"  verified: A @ v == lambda * v -> {verify_eigenpair(A, lam, v)}")
    print()

    # Pure scaling matrix (diagonal) — eigenvectors are the axes themselves
    B: Matrix = [
        [5, 0],
        [0, 2],
    ]
    print_matrix(B, "B (diagonal / pure scaling)")
    pairs_b = eigen_decompose_2x2(B)
    for lam, v in pairs_b:
        print(f"eigenvalue = {lam:.4f}   eigenvector = [{v[0]:.4f}, {v[1]:.4f}]")
    print()

    # Rotation matrix — complex eigenvalues, no real invariant direction
    theta = math.pi / 4
    R: Matrix = [
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta), math.cos(theta)],
    ]
    print_matrix(R, "R (45-degree rotation)")
    complex_eigvals = eigenvalues_2x2(R)
    print(f"eigenvalues = {complex_eigvals}  (complex -> no real invariant direction)")
