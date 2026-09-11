"""
Day 36 — SVD & PCA: Image Compression via Rank Reduction
NIZAM AI · 150-Day AI Engineering Protocol

Goal: use SVD (A = U * Sigma * V^T) to compress a grayscale image by
keeping only the top-k singular values, then compare visual quality
and storage size across k = 5, 20, 50.

Core intuition: SVD ranks the "directions" of a matrix by how much
information they carry (singular values, largest first). Keeping only
the top k means approximating A with a much lower-rank matrix — this
IS the math behind PCA (dimensionality reduction) and image/data
compression: most of the visual information lives in a handful of
dominant singular values.
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt


def compress_with_svd(image: np.ndarray, k: int) -> np.ndarray:
    """
    Reconstruct a rank-k approximation of `image` using its top-k
    singular values/vectors: A_k = U[:, :k] @ diag(S[:k]) @ Vt[:k, :]
    """
    U, S, Vt = np.linalg.svd(image, full_matrices=False)
    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]
    return U_k @ np.diag(S_k) @ Vt_k


def storage_ratio(image_shape: tuple[int, int], k: int) -> float:
    """
    Original storage: m * n floats.
    Compressed storage: k * (m + n + 1) floats (U_k columns + S_k + Vt_k rows).
    Returns the fraction of original size the compressed version uses.
    """
    m, n = image_shape
    original = m * n
    compressed = k * (m + n + 1)
    return compressed / original


def make_synthetic_image(size: int = 200, seed: int = 42) -> np.ndarray:
    """
    Build a synthetic grayscale image (no external file needed) that
    mixes smooth structure with high-frequency detail (concentric
    pattern + per-pixel noise), so low-rank approximation artifacts at
    small k are actually visible — a real photo behaves the same way:
    most energy sits in a few singular values, but fine detail needs more.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(-3, 3, size)
    y = np.linspace(-3, 3, size)
    xx, yy = np.meshgrid(x, y)
    structure = np.sin(xx ** 2 + yy ** 2) + 0.5 * np.cos(3 * xx) + 0.3 * xx
    noise = rng.normal(scale=0.15, size=(size, size))
    image = structure + noise
    # Normalize to 0-255 range like a real grayscale image
    image = (image - image.min()) / (image.max() - image.min()) * 255
    return image


def plot_compression_comparison(image: np.ndarray, ks: list[int], filename: str) -> None:
    """Side-by-side comparison: original + each rank-k reconstruction."""
    n_plots = len(ks) + 1
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4.5))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title(f"Original\n(rank {min(image.shape)})")
    axes[0].axis("off")

    for ax, k in zip(axes[1:], ks):
        reconstructed = compress_with_svd(image, k)
        ratio = storage_ratio(image.shape, k)
        ax.imshow(reconstructed, cmap="gray")
        ax.set_title(f"k={k}\n({ratio:.1%} of original size)")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    print(f"Saved comparison to {filename}")


if __name__ == "__main__":
    image = make_synthetic_image(size=200)
    ks = [5, 20, 50]

    print(f"Image shape: {image.shape}")
    for k in ks:
        ratio = storage_ratio(image.shape, k)
        reconstructed = compress_with_svd(image, k)
        error = np.linalg.norm(image - reconstructed) / np.linalg.norm(image)
        print(
            f"k={k:3d}  storage={ratio:6.1%}  "
            f"relative reconstruction error={error:.4f}"
        )

    plot_compression_comparison(image, ks, "svd_compression_comparison.png")
