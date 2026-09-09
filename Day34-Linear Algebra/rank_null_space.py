"""
Day 34 — Column Space, Rank, Null Space & Linear Systems
NIZAM AI · 150-Day AI Engineering Protocol

Goal: implement rank computation (via Gaussian elimination / row echelon
form) from scratch, then validate against NumPy's np.linalg.

Core intuition: the column space of A is the set of all possible outputs
A*x can reach — every linear combination of A's columns. The rank is the
true dimensionality of that space: how many columns are actually
independent (not combinations of the others). The null space is the
opposite question — which inputs collapse to the zero vector? If the
null space is trivial (just the zero vector), no information is lost
and the transformation is invertible. This directly connects to Day 33:
a singular matrix (det = 0) always has rank < n and a non-trivial null space.
"""

from __future__ import annotations
from typing import List
import copy

Matrix = List[List[float]]

TOLERANCE = 1e-9


def _to_row_echelon(m: Matrix) -> Matrix:
    """
    Reduce a matrix to row echelon form via Gaussian elimination
    with partial pivoting. Returns a new matrix (does not mutate input).
    """
    mat = copy.deepcopy(m)
    rows = len(mat)
    cols = len(mat[0]) if rows > 0 else 0
    pivot_row = 0

    for col in range(cols):
        if pivot_row >= rows:
            break

        # Partial pivoting: find the row with the largest value in this column
        max_row = max(
            range(pivot_row, rows), key=lambda r: abs(mat[r][col])
        )
        if abs(mat[max_row][col]) < TOLERANCE:
            continue  # no usable pivot in this column, move to next column

        mat[pivot_row], mat[max_row] = mat[max_row], mat[pivot_row]

        # Eliminate all entries below the pivot
        for r in range(pivot_row + 1, rows):
            factor = mat[r][col] / mat[pivot_row][col]
            for c in range(col, cols):
                mat[r][c] -= factor * mat[pivot_row][c]

        pivot_row += 1

    return mat


def rank(m: Matrix) -> int:
    """
    Compute the rank of a matrix: the number of non-zero rows after
    reducing to row echelon form (i.e., the number of independent
    directions the matrix's columns span).
    """
    echelon = _to_row_echelon(m)
    count = 0
    for row in echelon:
        if any(abs(val) > TOLERANCE for val in row):
            count += 1
    return count


def null_space_dimension(m: Matrix) -> int:
    """
    Rank-nullity theorem: for an m x n matrix,
    dim(null space) = n (number of columns) - rank(A).
    This tells us how many independent directions get collapsed to zero.
    """
    n_cols = len(m[0]) if m else 0
    return n_cols - rank(m)


def is_invertible(m: Matrix) -> bool:
    """A square matrix is invertible iff its rank equals its size (full rank)."""
    n = len(m)
    return rank(m) == n


def solve_linear_system(a: Matrix, b: List[float]) -> str:
    """
    Classify a linear system Ax = b by comparing rank(A) with
    rank([A | b]) (the augmented matrix):
      - rank(A) < rank([A|b])        -> no solution (inconsistent)
      - rank(A) == rank([A|b]) == n  -> unique solution
      - rank(A) == rank([A|b]) < n   -> infinitely many solutions
    """
    augmented = [row + [b[i]] for i, row in enumerate(a)]
    rank_a = rank(a)
    rank_aug = rank(augmented)
    n_unknowns = len(a[0])

    if rank_a < rank_aug:
        return "No solution (inconsistent system)"
    elif rank_a == n_unknowns:
        return "Unique solution (full rank)"
    else:
        return f"Infinitely many solutions (rank={rank_a} < {n_unknowns} unknowns)"


def print_matrix(m: Matrix, label: str = "") -> None:
    if label:
        print(f"{label}:")
    for row in m:
        print("  [" + ", ".join(f"{v:7.3f}" for v in row) + "]")
    print()


if __name__ == "__main__":
    # Full-rank matrix example
    A: Matrix = [
        [1, 2, 3],
        [0, 1, 4],
        [5, 6, 0],
    ]
    print_matrix(A, "A (full rank)")
    print(f"rank(A) = {rank(A)}")
    print(f"null space dimension = {null_space_dimension(A)}")
    print(f"invertible = {is_invertible(A)}\n")

    # Rank-deficient matrix (row 3 = row 1 + row 2)
    B: Matrix = [
        [1, 2, 3],
        [4, 5, 6],
        [5, 7, 9],  # dependent: row1 + row2
    ]
    print_matrix(B, "B (rank-deficient)")
    print(f"rank(B) = {rank(B)}")
    print(f"null space dimension = {null_space_dimension(B)}")
    print(f"invertible = {is_invertible(B)}\n")

    # Linear system examples
    b1 = [1, 2, 3]
    print(f"Ax = {b1} -> {solve_linear_system(A, b1)}")

    b2 = [1, 2, 3]
    print(f"Bx = {b2} -> {solve_linear_system(B, b2)}")
