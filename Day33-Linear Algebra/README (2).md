# Day 33 — Determinant & Inverse Matrix

Part of the **NIZAM AI · 150-Day AI Engineering Protocol**
Month 2 · Mathematics & Deep Learning · Block Day 3/7

## 📌 Concept

The determinant is a single number that tells you how much a linear
transformation scales **area** (2D) or **volume** (3D). If `det(A) == 0`,
the transformation collapses space into a lower dimension — information
is lost, so the transformation **cannot be undone**. That's exactly why
an inverse only exists when the determinant is nonzero.

The inverse matrix `A⁻¹` satisfies `A × A⁻¹ = I` (identity) — it undoes
A's transformation, mapping every point back to where it started.

Key intuitions:
- **Zero determinant → singular matrix** → no inverse exists.
- **Negative determinant** → the transformation flips orientation (mirror effect).
- 2×2 formula: `det(A) = ad - bc`; inverse: `(1/det) × [[d,-b],[-c,a]]`.
- 3×3 uses cofactor expansion + the adjugate matrix.

## 📂 Files

| File | Description |
|---|---|
| `determinant_inverse.py` | From-scratch determinant (2x2, 3x3 via cofactor expansion) and inverse (2x2, 3x3 via adjugate method) |
| `validate_with_numpy.py` | Validates both implementations against `np.linalg.det` / `np.linalg.inv` across fixed and random test cases |

## ▶️ Usage

```bash
# Run the from-scratch implementation with example output
python3 determinant_inverse.py

# Validate correctness against NumPy
pip install numpy
python3 validate_with_numpy.py
```

## ✅ Test Results

```
14/14 checks passed
```

Covers: fixed 2x2 and 3x3 matrices, multiple random invertible matrices,
correct error handling on singular (non-invertible) matrices, and
confirmation that `A @ A⁻¹ == Identity`.

## 📚 Sources

- [3Blue1Brown — Essence of Linear Algebra, ch. 5–7](https://www.3blue1brown.com/topics/linear-algebra)
- MIT 18.06 (Gilbert Strang)
- Khan Academy — Linear Algebra

## 🔁 Refactor Log

- Initial implementation: separate 2x2/3x3 functions with public dispatchers
- Added singular-matrix detection with descriptive error message
- Added NumPy cross-validation suite, including an identity-check sanity test

---
*NIZAM AI Day 33/150 · 21% overall progress*
