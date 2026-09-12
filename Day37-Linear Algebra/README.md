# Day 37 — Week 5 Review: Linear Algebra Cheat-Sheet & Exercises

Part of the **NIZAM AI · 150-Day AI Engineering Protocol**
Month 2 · Mathematics & Deep Learning · Block Day 7/7 (Week Review)

## 📌 Cheat-Sheet

| Topic | Core formula / idea |
|---|---|
| **Matrix multiplication** | `C[i][j] = Σ A[i][k]·B[k][j]` — linear transformation, not commutative |
| **Determinant** | `det = ad-bc` (2x2) — area/volume scale factor; 0 means the transformation collapses space |
| **Inverse** | `A·A⁻¹ = I`, exists only when `det ≠ 0` |
| **Rank / null space** | rank = number of independent directions; `dim(null) = n - rank` |
| **Eigenvalue/eigenvector** | `Av = λv` — invariant direction, λ = scale factor |
| **SVD / PCA** | `A = UΣVᵀ`, keeping top-k gives compression / dimensionality reduction |

These six ideas connect: **det=0 ⟺ rank<n ⟺ non-trivial null space ⟺ no inverse ⟺ at least one eigenvalue=0**.

## 📂 Files

| File | Description |
|---|---|
| `week5_review_exercises.py` | 10 worked exercises spanning Days 32–36, each with inline explanation of *why* |
| `test_exercises.py` | Automated assertion-based tests validating every exercise's core claim |

## ▶️ Usage

```bash
pip install numpy
python3 week5_review_exercises.py   # run all 10 worked exercises with explanations
python3 test_exercises.py           # run automated correctness checks
```

## ✅ Test Results

```
10/10 checks passed
```

## 🔗 What each exercise ties together

1. Matrix multiplication — order matters (Day 32)
2. Determinant as area scaling (Day 33)
3. Inverse solving `Ax=b` directly (Day 33)
4. Singular matrix: det=0 **and** rank<n together (Day 33 + 34)
5. Rank of a rectangular matrix (Day 34)
6. Classifying `Ax=b` — unique / infinite / no solution (Day 34)
7. Eigen-decomposition verified against the definition (Day 35)
8. Eigenvalue magnitude ⟶ growth/shrink over repeated steps — the same
   mechanism behind exploding/vanishing gradients in neural nets (Day 35)
9. SVD low-rank reconstruction error vs. k (Day 36)
10. SVD singular values *are* PCA's covariance eigenvalues — the direct
    link between Day 35 and Day 36 (Day 35 + 36)

## 📚 Sources

- 3Blue1Brown — Essence of Linear Algebra
- MIT 18.06 (Gilbert Strang)
- Khan Academy — Linear Algebra

---
*NIZAM AI Day 37/150 · Week 5 complete*
