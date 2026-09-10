"""
Day 35 — Visualizing Eigenvectors as Invariant Directions
NIZAM AI · 150-Day AI Engineering Protocol

Visualizes what makes eigenvectors special: applying the transformation
A to a grid of vectors around the unit circle, most vectors rotate AND
scale — but the eigenvectors only scale, staying on their own line.

Saves the plot to 'eigenvector_visualization.png' instead of calling
plt.show(), so it works in headless/non-interactive environments too.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
from eigen import eigen_decompose_2x2


def plot_transformation(m, filename: str = "eigenvector_visualization.png"):
    A = np.array(m, dtype=float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # --- Left plot: unit circle before transformation ---
    theta = np.linspace(0, 2 * np.pi, 100)
    circle = np.array([np.cos(theta), np.sin(theta)])

    ax1.plot(circle[0], circle[1], color="#4C72B0", label="Unit circle (before)")
    ax1.axhline(0, color="gray", linewidth=0.5)
    ax1.axvline(0, color="gray", linewidth=0.5)
    ax1.set_title("Before: unit circle")
    ax1.set_aspect("equal")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # --- Right plot: circle after transformation A, with eigenvectors highlighted ---
    transformed = A @ circle
    ax2.plot(transformed[0], transformed[1], color="#DD8452", label="Transformed by A")
    ax2.axhline(0, color="gray", linewidth=0.5)
    ax2.axvline(0, color="gray", linewidth=0.5)

    pairs = eigen_decompose_2x2(m)
    colors = ["#55A868", "#C44E52"]
    for (lam, v), color in zip(pairs, colors):
        v = np.array(v)
        # Draw the eigenvector line (both directions) to show it's invariant
        line = np.array([-v, v]) * 3
        ax2.plot(line[:, 0], line[:, 1], "--", color=color, alpha=0.6,
                  label=f"eigenvector (lambda={lam:.2f})")
        # Draw the actual scaled vector A @ v = lambda * v
        scaled = lam * v
        ax2.annotate(
            "", xy=(scaled[0], scaled[1]), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=color, linewidth=2),
        )

    ax2.set_title("After: A @ (unit circle) — eigenvectors stay on their line")
    ax2.set_aspect("equal")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    print(f"Saved visualization to {filename}")


if __name__ == "__main__":
    # Symmetric matrix: real, orthogonal eigenvectors — classic PCA-style example
    A = [[4, 1], [2, 3]]
    plot_transformation(A, "eigenvector_visualization.png")
