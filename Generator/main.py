from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import h5py


# ----------------------------
# Regex for filenames
# ----------------------------
RX_DILUTION = re.compile(
    r"^(?P<ID>\d{3})_"
    r"(?P<conc>\d+(?:\.\d+)?)_gL_"
    r"(?P<substance>[A-Za-z]+)_"
    r"(?P<power>\d+)mW_"
    r"(?P<integration>\d+)ms_"
    r"(?P<rep>\d+)rep_"
    r"data\.csv$"
)
RX_TURB = re.compile(r"^(?P<ID>\d+)_turbidity_background\.csv$")


# ----------------------------
# IO helpers
# ----------------------------
def read_csv_spectrum(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return (wavenumber, intensities) float32 arrays."""
    df = pd.read_csv(
        csv_path,
        usecols=["Wavenumber", "Intensities"],
        dtype={"Wavenumber": "float32", "Intensities": "float32"},
    )
    wn = df["Wavenumber"].to_numpy(dtype=np.float32, copy=False)
    y = df["Intensities"].to_numpy(dtype=np.float32, copy=False)
    return wn, y


def load_dilution_folder(folder: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load dilution spectra from CSVs.

    Returns:
      wn:   (1024,)
      X:    (n,1024)
      conc: (n,) concentration parsed from filename
    """
    rows = []
    for p in folder.glob("*.csv"):
        m = RX_DILUTION.match(p.name)
        if not m:
            continue
        d = m.groupdict()
        rows.append((int(d["ID"]), float(d["conc"]), p))

    if not rows:
        raise RuntimeError(f"No matching dilution CSVs found in: {folder}")

    rows.sort(key=lambda t: t[0])  # sort by ID
    conc = np.array([c for _, c, _ in rows], dtype=np.float32)
    files = [p for _, _, p in rows]

    wn0, y0 = read_csv_spectrum(files[0])
    L = y0.shape[0]
    if L != 1024:
        raise RuntimeError(f"{files[0].name} length {L}, expected 1024")

    X = np.empty((len(files), L), dtype=np.float32)
    X[0] = y0

    for i, p in enumerate(files[1:], start=1):
        wn, y = read_csv_spectrum(p)
        if y.shape[0] != L:
            raise RuntimeError(f"{p.name} length {y.shape[0]} != {L}")
        if not np.allclose(wn, wn0, rtol=0, atol=1e-3):
            raise RuntimeError(f"Wavenumber grid mismatch in {p.name}")
        X[i] = y

    return wn0, X, conc


def load_turbidity_folder(folder: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load turbidity background spectra.

    Returns:
      wn: (1024,)
      X:  (n,1024)
    """
    rows = []
    for p in folder.glob("*.csv"):
        m = RX_TURB.match(p.name)
        if not m:
            continue
        rows.append((int(m.group("ID")), p))

    if not rows:
        raise RuntimeError(f"No matching turbidity CSVs found in: {folder}")

    rows.sort(key=lambda t: t[0])  # sort by turbidity ID
    files = [p for _, p in rows]

    wn0, y0 = read_csv_spectrum(files[0])
    L = y0.shape[0]
    if L != 1024:
        raise RuntimeError(f"{files[0].name} length {L}, expected 1024")

    X = np.empty((len(files), L), dtype=np.float32)
    X[0] = y0

    for i, p in enumerate(files[1:], start=1):
        wn, y = read_csv_spectrum(p)
        if y.shape[0] != L:
            raise RuntimeError(f"{p.name} length {y.shape[0]} != {L}")
        if not np.allclose(wn, wn0, rtol=0, atol=1e-3):
            raise RuntimeError(f"Wavenumber grid mismatch in {p.name}")
        X[i] = y

    return wn0, X


def save_library(lib_dir: Path, name: str, wn: np.ndarray, X: np.ndarray, conc: Optional[np.ndarray] = None) -> None:
    lib_dir.mkdir(parents=True, exist_ok=True)
    np.save(lib_dir / f"{name}_wn.npy", wn.astype(np.float32))
    np.save(lib_dir / f"{name}_X.npy", X.astype(np.float32))
    if conc is not None:
        np.save(lib_dir / f"{name}_conc.npy", conc.astype(np.float32))


def load_library(lib_dir: Path, name: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    wn = np.load(lib_dir / f"{name}_wn.npy").astype(np.float32)
    X = np.load(lib_dir / f"{name}_X.npy").astype(np.float32)
    conc_path = lib_dir / f"{name}_conc.npy"
    conc = np.load(conc_path).astype(np.float32) if conc_path.exists() else None
    return wn, X, conc


# ----------------------------
# Fast block perturbations
# ----------------------------
def apply_peakshape_block(
    X_block: np.ndarray,
    wn: np.ndarray,
    rng: np.random.Generator,
    fwhm: float = 0.0,
    alpha: float = 0.0,
    beta: float = 0.0,
    jitter: float = 0.2,   # ±20% like your PeakShapeProcessor
) -> None:
    """
    In-place on X_block (...,1024).
    Uses scipy.ndimage.gaussian_filter1d for fast broadening if fwhm > 0.
    """
    if fwhm > 0 or alpha != 0 or beta != 0:
        pass
    else:
        return

    # Per-block jitter (fast). If you want per-spectrum jitter, tell me.
    fwhm_j = max(0.0, float(fwhm) * (1.0 + jitter * rng.standard_normal()))
    alpha_j = float(alpha) * (1.0 + jitter * rng.standard_normal())
    beta_j = float(beta) * (1.0 + jitter * rng.standard_normal())

    if fwhm_j > 0:
        try:
            from scipy.ndimage import gaussian_filter1d
        except Exception as e:
            raise RuntimeError("SciPy is required for broadening. Install: pip install scipy") from e

        dnu = float(np.abs(np.mean(np.diff(wn))))
        sigma = fwhm_j / 2.3548
        sigma_pts = sigma / dnu
        gaussian_filter1d(X_block, sigma=sigma_pts, axis=-1, mode="reflect", output=X_block)

    # Throughput tilt/curve (vectorized)
    nu_c = 0.5 * (float(wn.min()) + float(wn.max()))
    nu_range = float(wn.max() - wn.min())
    nu_norm = (wn - nu_c) / nu_range  # [-0.5,0.5]
    f = (1.0 + alpha_j * nu_norm + beta_j * (nu_norm ** 2)).astype(np.float32)
    X_block *= f


def apply_axis_perturb_block(
    X_block: np.ndarray,
    wn: np.ndarray,
    rng: np.random.Generator,
    shift_std: float = 1.0,
    warp_strength: float = 0.001,
    order: int = 3,
) -> np.ndarray:
    """
    Returns a NEW array with axis perturbation applied to the whole block.
    (Interpolation makes true in-place hard.)
    """
    try:
        from scipy.interpolate import interp1d
    except Exception as e:
        raise RuntimeError("SciPy is required for axis perturbation. Install: pip install scipy") from e

    X = X_block

    # Global shift
    delta = rng.normal(0.0, shift_std)
    wn_shifted = wn + delta

    f1 = interp1d(
        wn, X, kind="cubic", axis=-1,
        bounds_error=False,
        fill_value=(X[..., 0], X[..., -1]),
    )
    Xs = f1(wn_shifted).astype(np.float32, copy=False)

    # Polynomial warp (skip constant + linear)
    nu_bar = float(np.mean(wn))
    nu_range = float(wn.max() - wn.min())
    nu_norm = (wn - nu_bar) / nu_range

    coeffs = rng.normal(0.0, warp_strength, order + 1)
    coeffs[0] = 0.0
    coeffs[1] = 0.0
    delta_nu = (np.polyval(coeffs[::-1], nu_norm) * nu_range).astype(np.float32)
    wn_warped = wn + delta_nu

    f2 = interp1d(
        wn, Xs, kind="cubic", axis=-1,
        bounds_error=False,
        fill_value=(Xs[..., 0], Xs[..., -1]),
    )
    return f2(wn_warped).astype(np.float32, copy=False)


# ----------------------------
# HDF5 generator
# ----------------------------
def generate_all_sums_h5(
    BSA: np.ndarray,
    mAb: np.ndarray,
    MP: np.ndarray,
    turb: np.ndarray,
    wn: np.ndarray,
    mab_conc: np.ndarray,          # (nB,)
    out_h5: Path,
    apply_perturb: bool = True,
    seed: int = 123,
    with_labels: bool = False,
    fwhm: float = 6.0,
    alpha: float = 0.05,
    beta: float = 0.02,
    shift_std: float = 1.0,
    warp_strength: float = 0.001,
    compression: str = "lzf",      # "lzf", "gzip", or "none"
) -> None:
    """
    Writes:
      X[i]      = BSA[a] + mAb[b] + MP[c] + turb[d]
      y_mab[i]  = mab_conc[b]
      indices[i]= (d,a,b,c)  if with_labels

    H5 datasets:
      /X      float32 (N,1024)
      /y_mab  float32 (N,)
      /wn     float32 (1024,)
      /indices int32 (N,4) optional
    """
    BSA = np.asarray(BSA, dtype=np.float32)
    mAb = np.asarray(mAb, dtype=np.float32)
    MP = np.asarray(MP, dtype=np.float32)
    turb = np.asarray(turb, dtype=np.float32)
    wn = np.asarray(wn, dtype=np.float32)
    mab_conc = np.asarray(mab_conc, dtype=np.float32)

    nA, L = BSA.shape
    nB, _ = mAb.shape
    nC, _ = MP.shape
    nD, _ = turb.shape

    if mab_conc.shape[0] != nB:
        raise ValueError(f"mab_conc length {mab_conc.shape[0]} != nB {nB}")

    N = nA * nB * nC * nD
    block = nB * nC  # rows per (d,a)

    # Precompute mAb + MP once
    BC = np.empty((nB, nC, L), dtype=np.float32)
    np.add(mAb[:, None, :], MP[None, :, :], out=BC)

    # y for one (d,a) block (b changes slow, c fast)
    y_block = np.repeat(mab_conc, nC).astype(np.float32)

    # indices for one (d,a) block
    b_block = np.repeat(np.arange(nB, dtype=np.int32), nC)
    c_block = np.tile(np.arange(nC, dtype=np.int32), nB)

    rng = np.random.default_rng(seed)

    # H5 compression handling
    comp = None if compression == "none" else compression

    out_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_h5, "w") as f:
        # chunk rows: keep it not too huge, but aligned with our writes
        chunk_rows = min(block, 4096)
        dset_X = f.create_dataset(
            "X",
            shape=(N, L),
            dtype="float32",
            chunks=(chunk_rows, L),
            compression=comp,
        )
        dset_y = f.create_dataset(
            "y_mab",
            shape=(N,),
            dtype="float32",
            chunks=(min(block, 8192),),
            compression=comp,
        )
        f.create_dataset("wn", data=wn, dtype="float32")

        dset_idx = None
        if with_labels:
            dset_idx = f.create_dataset(
                "indices",
                shape=(N, 4),
                dtype="int32",
                chunks=(min(block, 8192), 4),
                compression=comp,
            )

        # metadata attrs
        f.attrs["nA_BSA"] = nA
        f.attrs["nB_mAb"] = nB
        f.attrs["nC_MP"] = nC
        f.attrs["nD_turb"] = nD
        f.attrs["seed"] = int(seed)
        f.attrs["apply_perturb"] = bool(apply_perturb)

        tmp = np.empty((nB, nC, L), dtype=np.float32)

        idx = 0
        for d in range(nD):
            td = turb[d]
            for a in range(nA):
                # tmp = BC + BSA[a] + turb[d]
                np.add(BC, BSA[a], out=tmp)
                np.add(tmp, td, out=tmp)

                if apply_perturb:
                    apply_peakshape_block(tmp, wn, rng, fwhm=fwhm, alpha=alpha, beta=beta)
                    warped = apply_axis_perturb_block(tmp, wn, rng, shift_std=shift_std, warp_strength=warp_strength)
                    tmp[:] = warped

                flat = tmp.reshape(block, L)
                dset_X[idx: idx + block, :] = flat
                dset_y[idx: idx + block] = y_block

                if dset_idx is not None:
                    blk = dset_idx[idx: idx + block]
                    blk[:, 0] = d
                    blk[:, 1] = a
                    blk[:, 2] = b_block
                    blk[:, 3] = c_block

                idx += block

        f.flush()

    print(f"Wrote HDF5: {out_h5}   X=({N},{L})  y_mab=({N},)")


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-cache", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--target", choices=["plastic", "ss", "both"], default="both")
    parser.add_argument("--with-labels", action="store_true", help="Store indices (d,a,b,c) inside the H5")
    parser.add_argument("--no-perturb", action="store_true", help="Disable peak/axis perturbations")
    parser.add_argument("--seed", type=int, default=123)

    # perturb params
    parser.add_argument("--fwhm", type=float, default=6.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--shift-std", type=float, default=1.0)
    parser.add_argument("--warp-strength", type=float, default=0.001)
    parser.add_argument("--compression", choices=["lzf", "gzip", "none"], default="lzf")

    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent  # .../Final_Synthesis
    data_root = root / "Cleaned_Data"
    lib_root = root / "Generator" / "cache"
    out_root = root / "Generator" / "outputs"

    folders = {
        "BSA": data_root / "BSA_dilution_wasatch",
        "MP": data_root / "MP_dilution_wasatch",
        "mAb_plastic": data_root / "mAb_dillution_plastic_synthetic",
        "mAb_ss": data_root / "mAb_dillution_SS_synthetic",
        "turb": data_root / "Turbidity_Backgrounds",
    }

    # Quick existence check
    for k, p in folders.items():
        if not p.exists():
            raise RuntimeError(f"Missing folder for {k}: {p}")

    if args.build_cache:
        wn_bsa, X_bsa, conc_bsa = load_dilution_folder(folders["BSA"])
        wn_mp, X_mp, conc_mp = load_dilution_folder(folders["MP"])
        wn_p, X_p, conc_p = load_dilution_folder(folders["mAb_plastic"])
        wn_s, X_s, conc_s = load_dilution_folder(folders["mAb_ss"])
        wn_t, X_t = load_turbidity_folder(folders["turb"])

        # sanity: all wavenumbers must match
        for name, wn in [("MP", wn_mp), ("mAb_plastic", wn_p), ("mAb_ss", wn_s), ("turb", wn_t)]:
            if not np.allclose(wn, wn_bsa, rtol=0, atol=1e-3):
                raise RuntimeError(f"Wavenumber grid mismatch between BSA and {name}")

        save_library(lib_root, "BSA", wn_bsa, X_bsa, conc_bsa)
        save_library(lib_root, "MP", wn_mp, X_mp, conc_mp)
        save_library(lib_root, "mAb_plastic", wn_p, X_p, conc_p)
        save_library(lib_root, "mAb_ss", wn_s, X_s, conc_s)
        save_library(lib_root, "turb", wn_t, X_t, None)

        print(f"Cache written to: {lib_root}")

    if args.generate:
        # require cache
        needed = ["BSA_wn.npy", "BSA_X.npy", "MP_X.npy", "turb_X.npy"]
        for fn in needed:
            if not (lib_root / fn).exists():
                raise RuntimeError(f"Cache missing ({fn}). Run: python main.py --build-cache")

        wn, BSA, _ = load_library(lib_root, "BSA")
        _, MP, _ = load_library(lib_root, "MP")
        _, turb, _ = load_library(lib_root, "turb")

        out_root.mkdir(parents=True, exist_ok=True)

        apply_perturb = not args.no_perturb

        if args.target in ("plastic", "both"):
            _, mAb_plastic, mab_conc_p = load_library(lib_root, "mAb_plastic")
            if mab_conc_p is None:
                raise RuntimeError("Missing mAb_plastic_conc.npy in cache. Re-run --build-cache.")
            out_h5 = out_root / "synthetic_plastic.h5"
            generate_all_sums_h5(
                BSA=BSA, mAb=mAb_plastic, MP=MP, turb=turb,
                wn=wn, mab_conc=mab_conc_p, out_h5=out_h5,
                apply_perturb=apply_perturb, seed=args.seed, with_labels=args.with_labels,
                fwhm=args.fwhm, alpha=args.alpha, beta=args.beta,
                shift_std=args.shift_std, warp_strength=args.warp_strength,
                compression=args.compression,
            )

        if args.target in ("ss", "both"):
            _, mAb_ss, mab_conc_s = load_library(lib_root, "mAb_ss")
            if mab_conc_s is None:
                raise RuntimeError("Missing mAb_ss_conc.npy in cache. Re-run --build-cache.")
            out_h5 = out_root / "synthetic_ss.h5"
            generate_all_sums_h5(
                BSA=BSA, mAb=mAb_ss, MP=MP, turb=turb,
                wn=wn, mab_conc=mab_conc_s, out_h5=out_h5,
                apply_perturb=apply_perturb, seed=args.seed, with_labels=args.with_labels,
                fwhm=args.fwhm, alpha=args.alpha, beta=args.beta,
                shift_std=args.shift_std, warp_strength=args.warp_strength,
                compression=args.compression,
            )


if __name__ == "__main__":
    main()
