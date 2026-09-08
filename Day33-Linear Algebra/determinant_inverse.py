"""
Day 33 — Determinant & Inverse Matrix
NIZAM AI · 150-Day AI Engineering Protocol

Goal: implement determinant (2x2, 3x3) and matrix inverse from scratch,
then validate against NumPy's np.linalg module.

Core intuition: the determinant measures how much a linear transformation
scales area (2D) or volume (3D). If det(A) == 0, the transformation
collapses space into a lower dimension — information is lost, so no
inverse exists (the matrix is "singular"). A negative determinant means
the transformation flips orientation (like a mirror).
"""

from __future__ import annotations
from typing import List

Matrix = List[List[float]]


def determinant_2x2(m: Matrix) -> float:
    """det([[a, b], [c, d]]) = ad - bc"""
    a, b = m[0]
    c, d = m[1]
    return a * d - b * c


def determinant_3x3(m: Matrix) -> float:
    """
    Cofactor expansion along the first row.
    det(A) = a(ei - fh) - b(di - fg) + c(dh - eg)
    for A = [[a, b, c], [d, e, f], [g, h, i]]
    """
    a, b, c = m[0]
    d, e, f = m[1]
    g, h, i = m[2]
    return (
        a * (e * i - f * h)
        - b * (d * i - f * g)
        + c * (d * h - e * g)
    )


def determinant(m: Matrix) -> float:
    """Dispatches to the 2x2 or 3x3 implementation based on matrix size."""
    n = len(m)
    if n == 2:
        return determinant_2x2(m)
    if n == 3:
        return determinant_3x3(m)
    raise NotImplementedError(
        f"determinant() only supports 2x2 and 3x3 matrices, got {n}x{n}"
    )


def inverse_2x2(m: Matrix) -> Matrix:
    """
    Inverse of [[a, b], [c, d]] = (1/det) * [[d, -b], [-c, a]]

    Raises:
        ValueError: if the matrix is singular (det == 0), since a
        transformation that collapses space cannot be undone.
    """
    det = determinant_2x2(m)
    if det == 0:
        raise ValueError(
            "Matrix is singular (det = 0) — no inverse exists. "
            "The transformation collapses space and information is lost."
        )
    a, b = m[0]
    c, d = m[1]
    inv_det = 1.0 / det
    return [
        [d * inv_det, -b * inv_det],
        [-c * inv_det, a * inv_det],
    ]


def inverse_3x3(m: Matrix) -> Matrix:
    """
    Inverse via the adjugate (cofactor transpose) method:
    A^-1 = (1/det(A)) * adj(A)
    """
    det = determinant_3x3(m)
    if det == 0:
        raise ValueError(
            "Matrix is singular (det = 0) — no inverse exists."
        )

    a, b, c = m[0]
    d, e, f = m[1]
    g, h, i = m[2]

    # Matrix of cofactors, then transposed directly into the adjugate layout
    cofactors = [
        [ (e * i - f * h), -(d * i - f * g),  (d * h - e * g)],
        [-(b * i - c * h),  (a * i - c * g), -(a * h - b * g)],
        [ (b * f - c * e), -(a * f - c * d),  (a * e - b * d)],
    ]
    # Transpose to get the adjugate
    adjugate = [[cofactors[j][i] for j in range(3)] for i in range(3)]

    inv_det = 1.0 / det
    return [[val * inv_det for val in row] for row in adjugate]


def inverse(m: Matrix) -> Matrix:
    """Dispatches to the 2x2 or 3x3 implementation based on matrix size."""
    n = len(m)
    if n == 2:
        return inverse_2x2(m)
    if n == 3:
        return inverse_3x3(m)
    raise NotImplementedError(
        f"inverse() only supports 2x2 and 3x3 matrices, got {n}x{n}"
    )


def print_matrix(m: Matrix, label: str = "") -> None:
    if label:
        print(f"{label}:")
    for row in m:
        print("  [" + ", ".join(f"{v:7.3f}" for v in row) + "]")
    print()


if __name__ == "__main__":
    # 2x2 example
    A: Matrix = [
        [4, 7],
        [2, 6],
    ]
    print_matrix(A, "A (2x2)")
    print(f"det(A) = {determinant(A)}\n")
    A_inv = inverse(A)
    print_matrix(A_inv, "A^-1")

    # 3x3 example
    B: Matrix = [
        [1, 2, 3],
        [0, 1, 4],
        [5, 6, 0],
    ]
    print_matrix(B, "B (3x3)")
    print(f"det(B) = {determinant(B)}\n")
    B_inv = inverse(B)
    print_matrix(B_inv, "B^-1")

    # Singular matrix example — should raise
    S: Matrix = [
        [1, 2],
        [2, 4],  # row 2 is a scalar multiple of row 1 -> collapses to a line
    ]
    print(f"det(S) = {determinant(S)}  (S is singular, no inverse)")
    try:
        inverse(S)
    except ValueError as e:
        print(f"Correctly raised: {e}")
