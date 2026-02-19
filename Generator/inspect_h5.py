# inspect_h5.py
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt


def h5_take_rows(dset, idx):
    """h5py requires sorted increasing indices for fancy indexing."""
    idx = np.asarray(idx, dtype=np.int64)
    idx = np.unique(idx)           # remove duplicates
    idx.sort()                     # increasing order
    return dset[idx, ...], idx


def main():
    parser = argparse.ArgumentParser(description="Inspect synthetic Raman HDF5 output")
    parser.add_argument("--file", type=str, default="outputs/synthetic_plastic.h5")
    parser.add_argument("--n-spectra", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hist-sample", type=int, default=200000)
    parser.add_argument("--check-sample", type=int, default=5000)
    args = parser.parse_args()

    h5_path = Path(args.file)
    if not h5_path.exists():
        h5_path = Path(__file__).resolve().parent / args.file
    if not h5_path.exists():
        raise FileNotFoundError(f"Could not find: {args.file}")

    rng = np.random.default_rng(args.seed)

    with h5py.File(h5_path, "r") as f:
        X = f["X"]
        y = f["y_mab"]
        wn = f["wn"][:]

        N, L = X.shape
        print(f"\nFile: {h5_path}")
        print(f"X shape: {X.shape}, dtype: {X.dtype}")
        print(f"y_mab shape: {y.shape}, dtype: {y.dtype}")
        print(f"wn shape: {wn.shape}, range: {float(wn[0])} -> {float(wn[-1])}")

        # ---------- Numeric sanity checks ----------
        n_check = min(args.check_sample, N)
        idx_check = rng.integers(0, N, size=n_check)

        Xs, idx_check_sorted = h5_take_rows(X, idx_check)
        ys = y[idx_check_sorted]

        print("\nSanity check (sample):")
        print("  NaN present:", bool(np.isnan(Xs).any()))
        print("  Inf present:", bool(np.isinf(Xs).any()))
        print("  X min/max (sample):", float(Xs.min()), float(Xs.max()))
        print("  y_mab min/max (sample):", float(ys.min()), float(ys.max()))
        print("  y_mab unique (sample, up to 20):", np.unique(np.asarray(ys))[:20])

        # ---------- Plot random spectra ----------
        k = min(args.n_spectra, N)
        idx_plot = rng.integers(0, N, size=k)

        # For plotting, just read each row one by one (k is small)
        plt.figure()
        for i in idx_plot:
            plt.plot(wn, X[int(i), :])
        plt.xlabel("Wavenumber")
        plt.ylabel("Intensity")
        plt.title(f"{h5_path.name}: {k} random spectra")
        plt.tight_layout()
        plt.show()

        print("\nPlotted spectra y_mab values:")
        for i in idx_plot:
            print(f"  row {int(i)} -> y_mab = {float(y[int(i)])}")

        # ---------- Plot y_mab histogram ----------
        n_hist = min(args.hist_sample, N)
        idx_hist = rng.integers(0, N, size=n_hist)

        # sort for h5py
        idx_hist = np.unique(idx_hist.astype(np.int64))
        idx_hist.sort()
        y_hist = np.asarray(y[idx_hist])

        plt.figure()
        plt.hist(y_hist, bins=50)
        plt.xlabel("mAb concentration (g/L)")
        plt.ylabel("Count (sampled)")
        plt.title(f"{h5_path.name}: y_mab distribution (sample n={len(idx_hist)})")
        plt.tight_layout()
        plt.show()

        # ---------- Optional indices dataset ----------
        if "indices" in f:
            ind = f["indices"]
            d, a, b, c = ind[int(idx_plot[0])]
            print("\nindices dataset present.")
            print("Example indices for first plotted row:", (int(d), int(a), int(b), int(c)))


if __name__ == "__main__":
    main()
