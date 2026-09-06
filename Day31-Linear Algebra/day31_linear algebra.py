"""
Vector class implemented from scratch (no NumPy).
Day 31 - Linear Algebra: vectors, linear combination, span, basis.
"""

import math


class Vector:
    """A simple n-dimensional vector supporting core linear algebra ops."""

    def __init__(self, components):
        self.components = list(components)
        self.dim = len(self.components)

    def __repr__(self):
        return f"Vector({self.components})"

    def __add__(self, other):
        self._check_dim(other)
        return Vector([a + b for a, b in zip(self.components, other.components)])

    def __sub__(self, other):
        self._check_dim(other)
        return Vector([a - b for a, b in zip(self.components, other.components)])

    def __mul__(self, scalar):
        """Scalar multiplication: v * k"""
        return Vector([a * scalar for a in self.components])

    __rmul__ = __mul__  # allow k * v as well

    def dot(self, other):
        """Dot product: v1 . v2 = sum(v1_i * v2_i)"""
        self._check_dim(other)
        return sum(a * b for a, b in zip(self.components, other.components))

    def norm(self):
        """Euclidean norm (magnitude): ||v|| = sqrt(v . v)"""
        return math.sqrt(self.dot(self))

    def angle_with(self, other, degrees=True):
        """Angle between two vectors using: cos(theta) = (v1 . v2) / (||v1|| * ||v2||)"""
        self._check_dim(other)
        denom = self.norm() * other.norm()
        if denom == 0:
            raise ValueError("Cannot compute angle with a zero vector.")
        cos_theta = max(-1.0, min(1.0, self.dot(other) / denom))  # clamp for float errors
        theta = math.acos(cos_theta)
        return math.degrees(theta) if degrees else theta

    def is_orthogonal_to(self, other, tol=1e-9):
        """Two vectors are orthogonal if their dot product is ~0."""
        return abs(self.dot(other)) < tol

    def normalize(self):
        """Return a unit vector (norm = 1) in the same direction."""
        n = self.norm()
        if n == 0:
            raise ValueError("Cannot normalize the zero vector.")
        return Vector([a / n for a in self.components])

    @staticmethod
    def linear_combination(vectors, scalars):
        """c1*v1 + c2*v2 + ... + cn*vn"""
        if len(vectors) != len(scalars):
            raise ValueError("vectors and scalars must have the same length.")
        result = Vector([0] * vectors[0].dim)
        for v, c in zip(vectors, scalars):
            result = result + (v * c)
        return result

    def _check_dim(self, other):
        if self.dim != other.dim:
            raise ValueError(f"Dimension mismatch: {self.dim} vs {other.dim}")


if __name__ == "__main__":
    v1 = Vector([1, 2])
    v2 = Vector([3, 4])

    print("v1 =", v1)
    print("v2 =", v2)
    print("v1 + v2 =", v1 + v2)
    print("v1 - v2 =", v1 - v2)
    print("3 * v1 =", 3 * v1)
    print("dot(v1, v2) =", v1.dot(v2))
    print("norm(v1) =", round(v1.norm(), 4))
    print("angle(v1, v2) =", round(v1.angle_with(v2), 2), "degrees")
    print("v1 normalized =", v1.normalize())

    # Linear combination example (span demonstration)
    e1 = Vector([1, 0])
    e2 = Vector([0, 1])
    combo = Vector.linear_combination([e1, e2], [2, 3])
    print("2*e1 + 3*e2 =", combo)  # should be [2, 3]

    # Orthogonality check
    a = Vector([1, 0])
    b = Vector([0, 5])
    print("a orthogonal to b?", a.is_orthogonal_to(b))