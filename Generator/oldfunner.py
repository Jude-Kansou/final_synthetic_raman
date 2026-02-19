from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
import h5py

import numpy as np
import pandas as pd
from helper import apply_axis_perturb_block, apply_peakshape_block


# ----------------------------
# Regex for dilution filenames
# ----------------------------
RX_DILUTION = re.compile(
    r"^(?P<ID>\d{3})_(?P<conc>\d+(?:\.\d+)?)_gL_(?P<substance>[A-Za-z]+)_(?P<power>\d+)mW_(?P<integration>\d+)ms_(?P<rep>\d+)rep_data\.csv$"
)
RX_TURB = re.compile(r"^(?P<ID>\d+)_turbidity_background\.csv$")


def read_csv_spectrum(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (wavenumber, intensities) as float32 arrays.
    Assumes columns: Wavenumber, Intensities
    """
    df = pd.read_csv(csv_path, usecols=["Wavenumber", "Intensities"])
    wn = df["Wavenumber"].to_numpy(dtype=np.float32, copy=False)
    y = df["Intensities"].to_numpy(dtype=np.float32, copy=False)
    return wn, y


def load_dilution_folder(folder: Path) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Loads all dilution CSVs in folder into:
      wn: (1024,)
      X:  (n_files, 1024)
      meta: list of dicts with parsed filename fields
    """
    rows = []
    files = []
    for p in folder.glob("*.csv"):
        m = RX_DILUTION.match(p.name)
        if not m:
            continue
        d = m.groupdict()
        d["ID"] = int(d["ID"])
        d["conc"] = float(d["conc"])
        d["power"] = int(d["power"])
        d["integration"] = int(d["integration"])
        d["rep"] = int(d["rep"])
        d["file"] = str(p)
        rows.append(d)
        files.append(p)

    if not files:
        raise RuntimeError(f"No matching dilution CSVs found in {folder}")

    # Sort by numeric ID for stable ordering
    rows_sorted = sorted(rows, key=lambda r: r["ID"])
    files_sorted = [Path(r["file"]) for r in rows_sorted]

    wn0, y0 = read_csv_spectrum(files_sorted[0])
    L = y0.shape[0]
    if L != 1024:
        raise RuntimeError(f"{files_sorted[0].name} has length {L}, expected 1024")

    X = np.empty((len(files_sorted), L), dtype=np.float32)
    X[0] = y0

    for i, p in enumerate(files_sorted[1:], start=1):
        wn, y = read_csv_spectrum(p)
        if y.shape[0] != L:
            raise RuntimeError(f"{p.name} length {y.shape[0]} != {L}")
        # Check wavenumber grid matches (tight tolerance)
        if not np.allclose(wn, wn0, rtol=0, atol=1e-3):
            raise RuntimeError(f"Wavenumber grid mismatch in {p.name}")
        X[i] = y

    return wn0, X, rows_sorted


def load_turbidity_folder(folder: Path) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Loads turbidity backgrounds:
      wn: (1024,)  (if present in file; assumed same format)
      X:  (n_files, 1024)
      meta: list of dicts
    """
    rows = []
    files = []
    for p in folder.glob("*.csv"):
        m = RX_TURB.match(p.name)
        if not m:
            continue
        d = m.groupdict()
        d["ID"] = int(d["ID"])
        d["file"] = str(p)
        rows.append(d)
        files.append(p)

    if not files:
        raise RuntimeError(f"No matching turbidity CSVs found in {folder}")

    rows_sorted = sorted(rows, key=lambda r: r["ID"])
    files_sorted = [Path(r["file"]) for r in rows_sorted]

    wn0, y0 = read_csv_spectrum(files_sorted[0])
    L = y0.shape[0]
    if L != 1024:
        raise RuntimeError(f"{files_sorted[0].name} has length {L}, expected 1024")

    X = np.empty((len(files_sorted), L), dtype=np.float32)
    X[0] = y0

    for i, p in enumerate(files_sorted[1:], start=1):
        wn, y = read_csv_spectrum(p)
        if y.shape[0] != L:
            raise RuntimeError(f"{p.name} length {y.shape[0]} != {L}")
        if not np.allclose(wn, wn0, rtol=0, atol=1e-3):
            raise RuntimeError(f"Wavenumber grid mismatch in {p.name}")
        X[i] = y

    return wn0, X, rows_sorted


def save_library(out_dir: Path, name: str, wn: np.ndarray, X: np.ndarray, meta: List[Dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{name}_wn.npy", wn)
    np.save(out_dir / f"{name}_X.npy", X)
    (out_dir / f"{name}_meta.json").write_text(json.dumps(meta, indent=2))


def load_library(lib_dir: Path, name: str) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    wn = np.load(lib_dir / f"{name}_wn.npy")
    X = np.load(lib_dir / f"{name}_X.npy")
    meta = json.loads((lib_dir / f"{name}_meta.json").read_text())
    return wn.astype(np.float32), X.astype(np.float32), meta


def generate_all_sums(
    BSA: np.ndarray,
    mAb: np.ndarray,
    MP: np.ndarray,
    turb: np.ndarray,
    wn: np.ndarray,                 # <-- add this
    out_path: Path,
    labels_path: Path | None = None,
) -> None:
    BSA = np.asarray(BSA, dtype=np.float32)
    mAb = np.asarray(mAb, dtype=np.float32)
    MP = np.asarray(MP, dtype=np.float32)
    turb = np.asarray(turb, dtype=np.float32)
    wn = np.asarray(wn, dtype=np.float32)

    nA, L = BSA.shape
    nB, _ = mAb.shape
    nC, _ = MP.shape
    nD, _ = turb.shape

    N = nA * nB * nC * nD
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(N, L))

    labels = None
    if labels_path is not None:
        labels = np.lib.format.open_memmap(labels_path, mode="w+", dtype=np.int32, shape=(N, 4))

    # Precompute mAb + MP once
    BC = np.empty((nB, nC, L), dtype=np.float32)
    np.add(mAb[:, None, :], MP[None, :, :], out=BC)

    rng = np.random.default_rng(123)   # <-- create ONCE here

    block = nB * nC
    idx = 0

    for d in range(nD):
        td = turb[d]
        for a in range(nA):
            chunk = out[idx: idx + block].reshape(nB, nC, L)

            np.add(BC, BSA[a], out=chunk)
            np.add(chunk, td, out=chunk)

            # In-place (good)
            apply_peakshape_block(chunk, wn, rng, fwhm=6.0, alpha=0.05, beta=0.02)

            # Returns new array -> copy back into memmap view
            warped = apply_axis_perturb_block(chunk, wn, rng, shift_std=1.0, warp_strength=0.001)
            chunk[:] = warped

            if labels is not None:
                lab = labels[idx: idx + block]
                lab[:, 0] = d
                lab[:, 1] = a
                lab[:, 2] = np.repeat(np.arange(nB, dtype=np.int32), nC)
                lab[:, 3] = np.tile(np.arange(nC, dtype=np.int32), nB)

            idx += block

    out.flush()
    if labels is not None:
        labels.flush()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-cache", action="store_true", help="Read CSVs and save .npy libraries")
    parser.add_argument("--generate", action="store_true", help="Generate synthetic sums using cached .npy libraries")
    parser.add_argument("--with-labels", action="store_true", help="Also write (d,a,b,c) index labels for each row")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent  # FINAL_SYNTHESIS
    data_root = root / "Cleaned_Data"
    lib_root = root / "Generator" / "cache"
    out_root = root / "Generator" / "outputs"
    out_root.mkdir(parents=True, exist_ok=True)

    folders = {
        "BSA": data_root / "BSA_dilution_wasatch",
        "MP": data_root / "MP_dilution_wasatch",
        "mAb_plastic": data_root / "mAb_dillution_plastic_synthetic",
        "mAb_ss": data_root / "mAb_dillution_SS_synthetic",
        "turb": data_root / "Turbidity_Backgrounds",
    }

    if args.build_cache:
        wn_bsa, X_bsa, meta_bsa = load_dilution_folder(folders["BSA"])
        wn_mp, X_mp, meta_mp = load_dilution_folder(folders["MP"])
        wn_p, X_p, meta_p = load_dilution_folder(folders["mAb_plastic"])
        wn_s, X_s, meta_s = load_dilution_folder(folders["mAb_ss"])
        wn_t, X_t, meta_t = load_turbidity_folder(folders["turb"])

        # sanity: all wavenumbers must match
        for name, wn in [("MP", wn_mp), ("mAb_plastic", wn_p), ("mAb_ss", wn_s), ("turb", wn_t)]:
            if not np.allclose(wn, wn_bsa, rtol=0, atol=1e-3):
                raise RuntimeError(f"Wavenumber grid mismatch between BSA and {name}")

        save_library(lib_root, "BSA", wn_bsa, X_bsa, meta_bsa)
        save_library(lib_root, "MP", wn_mp, X_mp, meta_mp)
        save_library(lib_root, "mAb_plastic", wn_p, X_p, meta_p)
        save_library(lib_root, "mAb_ss", wn_s, X_s, meta_s)
        save_library(lib_root, "turb", wn_t, X_t, meta_t)

        print(f"Cache written to: {lib_root}")

    if args.generate:
        wn_bsa, BSA, meta_bsa = load_library(lib_root, "BSA")
        wn_mp, MP, meta_mp = load_library(lib_root, "MP")
        wn_t, turb, meta_t = load_library(lib_root, "turb")

        # Plastic generation
        _, mAb_plastic, meta_p = load_library(lib_root, "mAb_plastic")
        plastic_out = out_root / "synthetic_plastic.npy"
        plastic_labels = out_root / "synthetic_plastic_labels.npy" if args.with_labels else None
        np.save(out_root / "wavenumber.npy", wn_bsa)  # same for all
        generate_all_sums(BSA, mAb_plastic, MP, turb, wn_bsa, plastic_out, plastic_labels)


        # SS generation
        _, mAb_ss, meta_s = load_library(lib_root, "mAb_ss")
        ss_out = out_root / "synthetic_ss.npy"
        ss_labels = out_root / "synthetic_ss_labels.npy" if args.with_labels else None
        generate_all_sums(BSA, mAb_ss, MP, turb, ss_out, ss_labels)


if __name__ == "__main__":
    main()
