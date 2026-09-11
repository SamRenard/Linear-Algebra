# Day 36 — SVD & PCA: Image Compression via Rank Reduction

Part of the **NIZAM AI · 150-Day AI Engineering Protocol**
Month 2 · Mathematics & Deep Learning · Block Day 6/7

## 📌 Concept

Every matrix `A` factors as `A = U Σ Vᵀ` (SVD), where Σ ranks the
matrix's "directions" by how much information they carry (singular
values, largest first). Keeping only the top `k` gives a low-rank
approximation that needs far less storage — this **is** the math
behind PCA and image/data compression.

## 📂 Files

| File | Description |
|---|---|
| `svd_compression.py` | Compresses a synthetic grayscale image via rank-k SVD reconstruction; compares k = 5, 20, 50 vs. original |
| `validate_svd.py` | Validates reconstruction correctness, confirms error shrinks as k grows, and proves SVD singular values equal PCA's covariance eigenvalues |
| `svd_compression_comparison.png` | Generated output — original vs. k=5/20/50 reconstructions side by side |

## ▶️ Usage

```bash
pip install numpy matplotlib
python3 svd_compression.py     # runs compression, saves comparison image
python3 validate_svd.py        # runs correctness + PCA-connection checks
```

## ✅ Results

```
k=  5   storage=  5.0%   relative error=0.0493
k= 20   storage= 20.1%   relative error=0.0428
k= 50   storage= 50.1%   relative error=0.0315
6/6 validation checks passed
```

Confirms: reconstruction matches manual SVD, full-rank recovers the
original exactly, error decreases monotonically with k, and — the key
Day 36 insight — **SVD singular values² / (n-1) equal the eigenvalues
of the data's covariance matrix**, directly connecting SVD to Day 35's
eigenvalues and to PCA.

## 📚 Sources

- 3Blue1Brown — Essence of Linear Algebra
- MIT 18.06 (Gilbert Strang)
- Khan Academy — Linear Algebra

## 🔁 Refactor Log

- Initial implementation: rank-k reconstruction via `U_k @ diag(S_k) @ Vt_k`
- Added storage-ratio calculation (k(m+n+1) vs. m·n floats)
- Added noise to the synthetic image so compression artifacts are visible at low k
- Added PCA-connection proof: SVD variance vs. covariance eigenvalues (Day 35 link)

---
*NIZAM AI Day 36/150*
