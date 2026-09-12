"""
Day 37 — Week 5 Review: 10 Linear Algebra Exercises
NIZAM AI · 150-Day AI Engineering Protocol

A cumulative practice set covering everything from this week:
  Day 32: Matrix multiplication
  Day 33: Determinant & inverse
  Day 34: Rank & null space
  Day 35: Eigenvalues & eigenvectors
  Day 36: SVD & PCA

Every exercise is solved with plain NumPy and includes an inline
explanation of *why* the result is what it is — the goal is fluency,
not just correct numbers.
"""

import numpy as np

SEP = "=" * 70


def header(title: str) -> None:
    print(f"\n{SEP}\n{title}\n{SEP}")


# ---------------------------------------------------------------------
# Exercise 1 — Matrix multiplication as transformation composition
# ---------------------------------------------------------------------
def exercise_1():
    header("Exercise 1: Matrix multiplication (Day 32)")
    A = np.array([[2, 0], [0, 3]])   # scales x by 2, y by 3
    B = np.array([[0, -1], [1, 0]])  # 90-degree rotation

    AB = A @ B
    BA = B @ A
    print("A (scale x2,y3):\n", A)
    print("B (90-deg rotation):\n", B)
    print("A @ B (rotate THEN scale):\n", AB)
    print("B @ A (scale THEN rotate):\n", BA)
    print(f"A@B == B@A? {np.array_equal(AB, BA)}  "
          f"-> order matters: rotating then scaling != scaling then rotating.")


# ---------------------------------------------------------------------
# Exercise 2 — Determinant as area scaling factor
# ---------------------------------------------------------------------
def exercise_2():
    header("Exercise 2: Determinant as area scale (Day 33)")
    A = np.array([[3, 1], [0, 2]])
    det = np.linalg.det(A)
    print(f"A:\n{A}\ndet(A) = {det}")
    print(f"-> the unit square (area 1) becomes a parallelogram of area {abs(det)}.")


# ---------------------------------------------------------------------
# Exercise 3 — Inverse solves a system directly
# ---------------------------------------------------------------------
def exercise_3():
    header("Exercise 3: Inverse matrix solves Ax=b (Day 33)")
    A = np.array([[2, 1], [1, 3]], dtype=float)
    b = np.array([5, 10], dtype=float)
    x = np.linalg.inv(A) @ b
    print(f"A:\n{A}\nb: {b}")
    print(f"x = A^-1 @ b = {x}")
    print(f"check A@x == b: {np.allclose(A @ x, b)}")


# ---------------------------------------------------------------------
# Exercise 4 — Singular matrix: no inverse, det=0
# ---------------------------------------------------------------------
def exercise_4():
    header("Exercise 4: Singular matrix (Day 33 + 34 link)")
    S = np.array([[2, 4], [1, 2]])  # row2 = 0.5 * row1 -> dependent
    det = np.linalg.det(S)
    rank = np.linalg.matrix_rank(S)
    print(f"S:\n{S}\ndet(S) = {det:.6f}  rank(S) = {rank}")
    print("-> det=0 and rank<n confirm S collapses the plane onto a line: no inverse.")


# ---------------------------------------------------------------------
# Exercise 5 — Rank of a rectangular matrix
# ---------------------------------------------------------------------
def exercise_5():
    header("Exercise 5: Rank of a rectangular matrix (Day 34)")
    M = np.array([[1, 2, 3], [2, 4, 6], [1, 0, 1]])  # row2 = 2*row1
    rank = np.linalg.matrix_rank(M)
    null_dim = M.shape[1] - rank
    print(f"M (3x3):\n{M}\nrank(M) = {rank}, dim(null space) = {null_dim}")
    print("-> row2 is a multiple of row1, so it adds no new direction.")


# ---------------------------------------------------------------------
# Exercise 6 — Ax=b classification (unique / infinite / none)
# ---------------------------------------------------------------------
def exercise_6():
    header("Exercise 6: Classifying Ax=b (Day 34)")
    A = np.array([[1, 2], [2, 4]])  # rank-deficient
    b_consistent = np.array([1, 2])   # b IS in the column space (2*row1)
    b_inconsistent = np.array([1, 0])  # b is NOT in the column space

    rank_a = np.linalg.matrix_rank(A)
    for b, label in [(b_consistent, "consistent b"), (b_inconsistent, "inconsistent b")]:
        aug = np.column_stack([A, b])
        rank_aug = np.linalg.matrix_rank(aug)
        if rank_a < rank_aug:
            verdict = "NO solution"
        elif rank_a == A.shape[1]:
            verdict = "UNIQUE solution"
        else:
            verdict = "INFINITE solutions"
        print(f"A rank={rank_a}, b={b} ({label}) -> rank([A|b])={rank_aug} -> {verdict}")


# ---------------------------------------------------------------------
# Exercise 7 — Eigenvalues/eigenvectors of a symmetric matrix
# ---------------------------------------------------------------------
def exercise_7():
    header("Exercise 7: Eigen-decomposition (Day 35)")
    A = np.array([[4, 1], [2, 3]], dtype=float)
    eigvals, eigvecs = np.linalg.eig(A)
    print(f"A:\n{A}\neigenvalues: {eigvals}")
    for i in range(len(eigvals)):
        v = eigvecs[:, i]
        print(f"  eigenvector {i}: {v}  ->  A@v = {A @ v}  vs  lambda*v = {eigvals[i]*v}")


# ---------------------------------------------------------------------
# Exercise 8 — Eigenvalues reveal stability (link to NN training)
# ---------------------------------------------------------------------
def exercise_8():
    header("Exercise 8: Eigenvalues & repeated transformation (Day 35 -> training stability)")
    A_stable = np.array([[0.5, 0], [0, 0.8]])    # eigenvalues < 1 -> shrinks
    A_unstable = np.array([[1.2, 0], [0, 0.9]])  # one eigenvalue > 1 -> grows

    v = np.array([1.0, 1.0])
    for name, M in [("stable (all |lambda|<1)", A_stable), ("unstable (one |lambda|>1)", A_unstable)]:
        eigvals = np.linalg.eigvals(M)
        result = v.copy()
        for _ in range(10):
            result = M @ result
        print(f"{name}: eigenvalues={eigvals}, after 10 applications ||v||={np.linalg.norm(result):.4f}")
    print("-> this is exactly why exploding/vanishing gradients depend on weight-matrix eigenvalues.")


# ---------------------------------------------------------------------
# Exercise 9 — SVD low-rank approximation
# ---------------------------------------------------------------------
def exercise_9():
    header("Exercise 9: SVD rank reduction (Day 36)")
    rng = np.random.default_rng(0)
    M = rng.normal(size=(6, 4))
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    print(f"M shape: {M.shape}, singular values: {np.round(S, 3)}")
    for k in [1, 2, 4]:
        approx = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
        err = np.linalg.norm(M - approx) / np.linalg.norm(M)
        print(f"  k={k}: relative reconstruction error = {err:.4f}")


# ---------------------------------------------------------------------
# Exercise 10 — SVD singular values <-> PCA variance (ties Day 35 + 36)
# ---------------------------------------------------------------------
def exercise_10():
    header("Exercise 10: SVD <-> PCA variance connection (Day 35 + 36)")
    rng = np.random.default_rng(1)
    data = rng.normal(size=(50, 3)) @ rng.normal(size=(3, 3))
    centered = data - data.mean(axis=0)

    _, S, _ = np.linalg.svd(centered, full_matrices=False)
    svd_variance = (S ** 2) / (data.shape[0] - 1)

    cov = np.cov(centered, rowvar=False)
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]

    print(f"SVD-derived variance:        {np.round(svd_variance, 4)}")
    print(f"Covariance-matrix eigenvals: {np.round(eigvals, 4)}")
    print(f"Match: {np.allclose(svd_variance, eigvals)}  "
          f"-> PCA IS eigen-decomposition of the covariance matrix, computed via SVD.")


if __name__ == "__main__":
    exercises = [
        exercise_1, exercise_2, exercise_3, exercise_4, exercise_5,
        exercise_6, exercise_7, exercise_8, exercise_9, exercise_10,
    ]
    for ex in exercises:
        ex()
    print(f"\n{SEP}\nAll 10 exercises completed.\n{SEP}")
