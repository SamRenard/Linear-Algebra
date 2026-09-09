# Day 34 — Column Space, Rank, Null Space & Linear Systems

Part of the **NIZAM AI · 150-Day AI Engineering Protocol**
Month 2 · Mathematics & Deep Learning · Block Day 4/7

## 📌 Concept

The **column space** of a matrix A is every output A×x can possibly
reach — all linear combinations of A's columns. The **rank** is the true
dimensionality of that space: how many columns are actually independent
(none is a combination of the others). If rank < number of columns,
information is lost — this directly connects to Day 33: a singular
matrix (det = 0) always has rank less than full and a non-trivial
null space.

The **null space** asks the opposite question: which inputs collapse to
the zero vector after the transformation? A trivial null space (only the
zero vector) means no information is lost — the transformation is
invertible.

Key intuitions:
- **Rank-nullity theorem**: `dim(null space) = n_columns - rank(A)`.
- A square matrix is **invertible iff it has full rank**.
- For `Ax = b`: compare `rank(A)` vs `rank([A|b])` to classify the
  system as having no solution, a unique solution, or infinitely many.

## 📂 Files

| File | Description |
|---|---|
| `rank_null_space.py` | From-scratch rank (via Gaussian elimination / row echelon form with partial pivoting), null space dimension, invertibility check, and linear system classifier |
| `validate_with_numpy.py` | Validates against `np.linalg.matrix_rank` across fixed and random matrices, plus sanity checks for invertibility and system solvability |

## ▶️ Usage

```bash
# Run the from-scratch implementation with example output
python3 rank_null_space.py

# Validate correctness against NumPy
pip install numpy
python3 validate_with_numpy.py
```

## ✅ Test Results

```
13/13 checks passed
```

Covers: full-rank and rank-deficient square matrices, a rectangular
matrix with dependent rows, random matrices, invertibility checks, and
`Ax = b` classification (unique / infinite / no solution).

## 📚 Sources

- [3Blue1Brown — Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra)
- MIT 18.06 (Gilbert Strang) — Lectures 6–8
- Khan Academy — Linear Algebra

## 🔁 Refactor Log

- Initial implementation: Gaussian elimination with partial pivoting for numerical stability
- Added rank-nullity theorem helper (`null_space_dimension`)
- Added `Ax = b` system classifier comparing `rank(A)` vs `rank([A|b])`
- Added NumPy cross-validation suite, including invertibility and solvability sanity checks

---
*NIZAM AI Day 34/150 · 21% overall progress*
