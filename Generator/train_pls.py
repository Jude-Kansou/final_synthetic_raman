# train_pls.py
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error


def sample_contiguous_blocks(N: int, n_samples: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample indices using contiguous blocks so h5py reads are fast and valid.
    Returns sorted indices.
    """
    if n_samples >= N:
        return np.arange(N, dtype=np.int64)

    block_len = max(1, int(block_len))
    n_blocks = int(np.ceil(n_samples / block_len))
    starts = rng.integers(0, max(1, N - block_len), size=n_blocks)

    idx = np.concatenate([np.arange(s, min(N, s + block_len), dtype=np.int64) for s in starts])
    idx = np.unique(idx)
    if idx.size > n_samples:
        idx = idx[:n_samples]
    idx.sort()
    return idx


def main():
    parser = argparse.ArgumentParser(description="Train/evaluate PLS on synthetic HDF5")
    parser.add_argument("--file", type=str, default="outputs/synthetic_plastic.h5")
    parser.add_argument("--n-train", type=int, default=150000)
    parser.add_argument("--n-test", type=int, default=50000)
    parser.add_argument("--block-len", type=int, default=4096, help="contiguous read block length")
    parser.add_argument("--components", type=int, default=12, help="PLS n_components")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    h5_path = Path(args.file)
    if not h5_path.exists():
        h5_path = Path(__file__).resolve().parent / args.file
    if not h5_path.exists():
        raise FileNotFoundError(f"Could not find: {args.file}")

    rng = np.random.default_rng(args.seed)

    with h5py.File(h5_path, "r") as f:
        Xd = f["X"]
        yd = f["y_mab"]

        N, L = Xd.shape
        print(f"File: {h5_path}")
        print(f"X: {Xd.shape} {Xd.dtype}   y: {yd.shape} {yd.dtype}")

        # Sample train/test indices as contiguous blocks (fast + valid)
        idx_train = sample_contiguous_blocks(N, min(args.n_train, N), args.block_len, rng)

        # Ensure test doesn't overlap train
        # Sample more then remove overlaps
        idx_test = sample_contiguous_blocks(N, min(args.n_test, N), args.block_len, rng)
        mask = np.isin(idx_test, idx_train, invert=True)
        idx_test = idx_test[mask]
        if idx_test.size < min(args.n_test, N):
            # top up with a contiguous chunk near the end
            extra = np.arange(max(0, N - (args.n_test - idx_test.size)), N, dtype=np.int64)
            extra = extra[~np.isin(extra, idx_train)]
            idx_test = np.unique(np.concatenate([idx_test, extra]))[: min(args.n_test, N)]
            idx_test.sort()

        print(f"Train samples: {idx_train.size}, Test samples: {idx_test.size}")

        # Load subsets into RAM (float32 is fine; sklearn will cast internally)
        X_train = np.asarray(Xd[idx_train, :], dtype=np.float32)
        y_train = np.asarray(yd[idx_train], dtype=np.float32)

        X_test = np.asarray(Xd[idx_test, :], dtype=np.float32)
        y_test = np.asarray(yd[idx_test], dtype=np.float32)

    # Model: scale then PLS
    model = Pipeline([
        ("scale", StandardScaler(with_mean=True, with_std=True)),
        ("pls", PLSRegression(n_components=args.components)),
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test).ravel()

    r2 = r2_score(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    print(f"\nPLS n_components={args.components}")
    print(f"Test R²:   {r2:.5f}")
    print(f"Test RMSE: {rmse:.5f} g/L")

    # ---- Plots ----
    plt.figure()
    plt.scatter(y_test, y_pred, s=6)
    plt.xlabel("True mAb (g/L)")
    plt.ylabel("Predicted mAb (g/L)")
    plt.title(f"PLS Predictions (R²={r2:.4f}, RMSE={rmse:.4f})")
    plt.tight_layout()
    plt.show()

    resid = y_pred - y_test
    plt.figure()
    plt.hist(resid, bins=60)
    plt.xlabel("Residual (pred - true) g/L")
    plt.ylabel("Count")
    plt.title("Residual distribution")
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.scatter(y_test, resid, s=6)
    plt.axhline(0, linewidth=1)
    plt.xlabel("True mAb (g/L)")
    plt.ylabel("Residual (g/L)")
    plt.title("Residuals vs True")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
