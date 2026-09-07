"""
Day 32 — Matrix Multiplication as Linear Transformation
NIZAM AI · 150-Day AI Engineering Protocol

Goal: implement matrix multiplication from scratch (triple nested loop),
then validate the result against NumPy.

Core intuition: multiplying A (m x n) by B (n x p) applies the linear
transformation encoded by A to every column vector of B. Each output
entry C[i][j] is the dot product of row i of A and column j of B —
i.e., a projection of that column onto the i-th transformation direction.
"""

from __future__ import annotations
from typing import List

Matrix = List[List[float]]


def matmul(a: Matrix, b: Matrix) -> Matrix:
    """
    Multiply matrix `a` (m x n) by matrix `b` (n x p) from scratch.

    Raises:
        ValueError: if inner dimensions don't match (n_cols(a) != n_rows(b)).
    """
    rows_a, cols_a = len(a), len(a[0])
    rows_b, cols_b = len(b), len(b[0])

    if cols_a != rows_b:
        raise ValueError(
            f"Dimension mismatch: A is {rows_a}x{cols_a}, "
            f"B is {rows_b}x{cols_b}. Inner dimensions must match "
            f"(cols_a == rows_b)."
        )

    # Initialize result matrix (m x p) with zeros
    result: Matrix = [[0.0 for _ in range(cols_b)] for _ in range(rows_a)]

    for i in range(rows_a):          # each row of A
        for j in range(cols_b):      # each column of B
            total = 0.0
            for k in range(cols_a):  # dot product across the shared dimension
                total += a[i][k] * b[k][j]
            result[i][j] = total

    return result


def print_matrix(m: Matrix, label: str = "") -> None:
    if label:
        print(f"{label}:")
    for row in m:
        print("  [" + ", ".join(f"{v:6.2f}" for v in row) + "]")
    print()


if __name__ == "__main__":
    A: Matrix = [
        [1, 2, 3],
        [4, 5, 6],
    ]
    B: Matrix = [
        [7, 8],
        [9, 10],
        [11, 12],
    ]

    print_matrix(A, "A (2x3)")
    print_matrix(B, "B (3x2)")

    C = matmul(A, B)
    print_matrix(C, "C = A x B (2x2)")

    # Demonstrate non-commutativity intuition: A x B != B x A (when both defined)
    D: Matrix = [
        [1, 2],
        [3, 4],
    ]
    E: Matrix = [
        [0, 1],
        [1, 0],
    ]
    print_matrix(matmul(D, E), "D x E")
    print_matrix(matmul(E, D), "E x D  (different! matrix mult is NOT commutative)")
