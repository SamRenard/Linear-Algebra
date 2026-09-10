# Day 35 — Eigenvalues & Eigenvectors

Part of the **NIZAM AI · 150-Day AI Engineering Protocol**
Month 2 · Mathematics & Deep Learning · Block Day 5/7

## 📌 Concept

If a matrix is a transformation, an **eigenvector** is one of the special
directions that doesn't get rotated by it — it only gets stretched or
squished (or flipped, but stays on the same line). The **eigenvalue** is
that stretch factor: `A @ v = λ * v` says that multiplying by A is, along
this direction, exactly the same as multiplying by a single number λ.

This matters because it turns a complicated transformation into a simple
one, viewed along the right axes. Real applications:
- **PCA** finds directions of maximum variance via the eigenvectors of a covariance matrix.
- **PageRank** finds the dominant eigenvector of a link-graph matrix.
- **Deep learning stability** (exploding/vanishing gradients) is analyzed via weight-matrix eigenvalues.

Eigenvalues come from solving the **characteristic equation**
`det(A - λI) = 0`.

## 📂 Files

| File | Description |
|---|---|
| `eigen.py` | From-scratch 2x2 eigenvalue solver (quadratic characteristic equation, handles complex eigenvalues for rotations) and eigenvector finder |
| `validate_with_numpy.py` | Validates eigenvalues and eigenvectors against `np.linalg.eig` across fixed and random symmetric matrices |
| `visualize_eigenvectors.py` | Matplotlib visualization: applies a transformation to the unit circle and highlights the eigenvector directions that stay invariant |
| `eigenvector_visualization.png` | Generated output — before/after transformation with eigenvector directions overlaid |

## ▶️ Usage

```bash
# Run the from-scratch implementation with example output
python3 eigen.py

# Validate correctness against NumPy
pip install numpy
python3 validate_with_numpy.py

# Generate the visualization (saves a PNG, no display needed)
pip install matplotlib
python3 visualize_eigenvectors.py
```

## ✅ Test Results

```
14/14 checks passed
```

Covers: an asymmetric matrix with real eigenvalues, a diagonal (pure
scaling) matrix, a symmetric matrix (PCA-style), random symmetric
matrices, plus verification that `A @ v == λ * v` for every computed
pair and that eigenvector directions match NumPy's.

## 📚 Sources

- [3Blue1Brown — Essence of Linear Algebra, ch. 14](https://www.3blue1brown.com/topics/linear-algebra)
- MIT 18.06 (Gilbert Strang)
- Khan Academy — Linear Algebra

## 🔁 Refactor Log

- Initial implementation: 2x2 characteristic equation via quadratic formula, complex-eigenvalue-aware
- Added `eigenvector_for_real_eigenvalue` handling degenerate (diagonal, zero-off-diagonal) cases
- Added `verify_eigenpair` sanity check (`A @ v == λ * v`)
- Added matplotlib visualization comparing before/after transformation with eigenvector overlay

---
*NIZAM AI Day 35/150 · 23% overall progress*
