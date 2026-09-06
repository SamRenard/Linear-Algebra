# Linear Algebra

A from-scratch implementation of core linear algebra concepts in pure Python — no NumPy, no shortcuts. Built as part of a structured AI engineering curriculum to build real intuition for the math behind machine learning.

## Why from scratch?

Libraries like NumPy hide the mechanics behind vector operations. Implementing them manually forces a real understanding of *why* the math works, not just *how* to call a function.

## What's inside

### `vector.py`

A custom `Vector` class supporting:

- **Vector addition & subtraction** — `v1 + v2`, `v1 - v2`
- **Scalar multiplication** — `3 * v1`
- **Dot product** — `v1.dot(v2)`
- **Norm (magnitude)** — `v1.norm()`, computed as `sqrt(v · v)`
- **Angle between vectors** — `v1.angle_with(v2)`, using `cos(θ) = (v1 · v2) / (‖v1‖ ‖v2‖)`
- **Orthogonality check** — `v1.is_orthogonal_to(v2)`
- **Normalization** — `v1.normalize()`, returns a unit vector
- **Linear combinations** — `Vector.linear_combination(vectors, scalars)`, demonstrating the building block of **span** and **basis**

## Example

```python
from vector import Vector

v1 = Vector([1, 2])
v2 = Vector([3, 4])

print(v1 + v2)              # Vector([4, 6])
print(v1.dot(v2))           # 11
print(v1.norm())            # 2.236...
print(v1.angle_with(v2))    # 10.3 (degrees)

# Linear combination — the foundation of span
e1 = Vector([1, 0])
e2 = Vector([0, 1])
print(Vector.linear_combination([e1, e2], [2, 3]))  # Vector([2, 3])
```

## Concepts covered

| Concept | Description |
|---|---|
| **Vector** | An ordered list of numbers representing magnitude and direction |
| **Linear combination** | `c1*v1 + c2*v2 + ... + cn*vn` — scaling and adding vectors together |
| **Span** | The set of all points reachable through linear combinations of a set of vectors |
| **Basis** | A minimal, linearly independent set of vectors that spans a space |

## Running

```bash
python3 vector.py
```

Runs a built-in demonstration of every method with printed output.

## Roadmap

- [ ] Matrix class (multiplication, transpose, determinant)
- [ ] Linear system solver (Gaussian elimination)
- [ ] Eigenvalues & eigenvectors from scratch
- [ ] Basis and rank checker

## Learning resources

- [3Blue1Brown — Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- [MIT 18.06 — Linear Algebra (Gilbert Strang)](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/)
- [Khan Academy — Linear Algebra](https://www.khanacademy.org/math/linear-algebra)

---

Part of a 150-day, 5-month AI engineering protocol.
