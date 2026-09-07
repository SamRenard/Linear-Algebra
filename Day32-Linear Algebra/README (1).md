# Day 32 — Matrix Multiplication as Linear Transformation

Part of the **NIZAM AI · 150-Day AI Engineering Protocol**
Month 2 · Mathematics & Deep Learning · Block Day 2/7

## 📌 Concept

Matrix multiplication is not just a numeric procedure — it's a **linear
transformation**. Multiplying `A × B` applies A's transformation to every
column vector of B. The dot-product rule (row · column) exists because each
output entry is a projection of an input vector onto a transformation
direction.

Key intuitions:
- **Not commutative**: `A × B ≠ B × A` in general — order of transformations matters.
- **Dimension rule**: `(m×n) @ (n×p) = (m×p)` — inner dimensions must match.
- This is exactly what a neural network layer does: `output = W @ input`.

## 📂 Files

| File | Description |
|---|---|
| `matrix_multiplication.py` | From-scratch matrix multiplication (triple nested loop), no libraries |
| `validate_with_numpy.py` | Validates the from-scratch implementation against NumPy (`np.allclose`) across fixed and random test cases |

## ▶️ Usage

```bash
# Run the from-scratch implementation with example output
python3 matrix_multiplication.py

# Validate correctness against NumPy
pip install numpy
python3 validate_with_numpy.py
```

## ✅ Test Results

```
7/7 checks passed
```

Covers: fixed known matrices, square matrices, multiple random shapes,
and correct error handling on dimension mismatch.

## 📚 Sources

- [3Blue1Brown — Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra)
- MIT 18.06 (Gilbert Strang)
- Khan Academy — Linear Algebra

## 🔁 Refactor Log

- Initial implementation: triple nested loop, O(m·n·p)
- Added dimension-mismatch validation with descriptive error message
- Added NumPy cross-validation suite for correctness confidence

---
*NIZAM AI Day 32/150 · 21% overall progress*
