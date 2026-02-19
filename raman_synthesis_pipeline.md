# Raman Synthetic Dataset Pipeline (Final_Synthesis)

This document describes the full pipeline used in this project: how the cleaned Raman CSV data is organized, how we parse labels (especially **mAb concentration**) from filenames, how we generate large synthetic datasets by **additive mixing**, how we apply **axis / peak-shape perturbations**, how to **inspect** the generated datasets, and how to **train a PLS regression model** to quantify dataset difficulty.

---

## 1) Project layout

```
Final_Synthesis/
├─ Cleaned_Data/
│  ├─ BSA_dilution_wasatch/
│  ├─ MP_dilution_wasatch/
│  ├─ mAb_dillution_plastic_synthetic/
│  ├─ mAb_dillution_SS_synthetic/
│  └─ Turbidity_Backgrounds/
└─ Generator/
   ├─ main.py
   ├─ inspect_h5.py
   ├─ train_pls.py
   └─ cache/            (created by --build-cache)
      outputs/          (created by --generate)
```

---

## 2) Cleaned data design

### 2.1 CSV format

Each Raman CSV contains **1024** points with columns:

- `Wavenumber`
- `Intensities`

The spectra are interpolated onto an equidistant axis (1024 points).  
All spectra **must share the same wavenumber grid** for the additive synthesis to be valid.

### 2.2 File naming convention (dilution folders)

All dilution datasets (BSA, MP, mAb plastic, mAb SS) follow:

```
{ID}_{Concentration}_gL_{Substance}_{Power}mW_{Integration}ms_{Repetitions}rep_data.csv
```

Example:
- `000_2_gL_BSA_450mW_8000ms_5rep_data.csv`
- `031_30_gL_MP_450mW_8000ms_5rep_data.csv`
- `500_0.21_gL_mAb_450mW_8000ms_5rep_data.csv`

The pipeline uses filename parsing to extract concentration and metadata.  
**No JSON metadata is required**.

### 2.3 File naming convention (turbidity backgrounds)

Turbidity background spectra follow:

```
{ID}_turbidity_background.csv
```

Example:
- `90_turbidity_background.csv`

These files are treated as **background components** in the additive mixing.

---

## 3) Parsing labels with Regex

The generator parses dilution filenames with a regex pattern:

- Extracts:
  - `ID` (integer)
  - `conc` (float)
  - `substance` (e.g., BSA, MP, mAb)
  - `power`, `integration`, `rep` (integers)

The turbidity background filenames use a separate regex to extract `ID`.

**mAb concentration (`y_mab`) is taken directly from `conc` parsed from the mAb filenames.**

---

## 4) Pipeline overview

There are two main stages:

1. **Cache stage (`--build-cache`)**
   - Read many CSVs once
   - Validate wavenumber alignment across files
   - Store fast-loading binary arrays in `Generator/cache/`

2. **Generate stage (`--generate`)**
   - Load cached libraries (`.npy`)
   - Generate synthetic spectra by additive mixing
   - Apply perturbations (peak-shape + axis)
   - Write the final dataset to `.h5` in `Generator/outputs/`

---

## 5) Cache stage (`--build-cache`)

### 5.1 What is cached

For each dilution folder (BSA, MP, mAb_plastic, mAb_ss), the cache writes:

- `{name}_X.npy` — float32 matrix of intensities, shape `(n_files, 1024)`
- `{name}_wn.npy` — float32 wavenumber axis, shape `(1024,)`
- `{name}_conc.npy` — float32 concentration array, shape `(n_files,)`  
  (This is what we later use to produce `y_mab`.)

For turbidity backgrounds:

- `turb_X.npy`, `turb_wn.npy` (no concentration file)

### 5.2 Why caching matters

CSV parsing is slow. Once cached, re-running generation is fast because:

- `.npy` loads are extremely fast
- The generator never re-reads CSVs during generation

### 5.3 Command

From `Final_Synthesis/Generator`:

```bash
python main.py --build-cache
```

After success, verify:

```bash
ls -lh cache
```

---

## 6) Synthetic generation stage (`--generate`)

### 6.1 Additive mixing model

Each synthetic spectrum is a simple sum of four components:

\[
S = S_{\text{BSA}}(a)\;+\;S_{\text{mAb}}(b)\;+\;S_{\text{MP}}(c)\;+\;S_{\text{turb}}(d)
\]

where:
- `a` indexes a BSA spectrum
- `b` indexes an mAb spectrum (plastic or SS)
- `c` indexes an MP spectrum
- `d` indexes a turbidity background spectrum

### 6.2 Efficient implementation (vectorized blocks)

To avoid slow nested Python loops across millions of combinations, we generate in blocks:

1. Precompute:
   - `BC[b,c,:] = mAb[b,:] + MP[c,:]`  → shape `(nB, nC, 1024)`
2. For each turbidity `d` and each BSA `a`:
   - `tmp = BC + BSA[a] + turb[d]`  → shape `(nB, nC, 1024)`
3. Apply perturbations to `tmp`
4. Write `tmp.reshape(nB*nC, 1024)` to disk

This uses only `nD * nA` Python iterations (e.g., `5 * 23 = 115`) and keeps the inner work in optimized NumPy/SciPy code.

---

## 7) Perturbations applied

We intentionally introduce distortions to break perfect linearity and make the dataset more realistic / harder for downstream regression.

### 7.1 Peak-shape perturbations (blockwise)

Applied to intensity values (not to the axis), typically in-place.

1. **Gaussian broadening**
   - Convolves spectra with a Gaussian kernel
   - Controlled by `--fwhm` (full width at half maximum)

2. **Throughput tilt**
   - Multiply by a linear factor across wavenumber axis
   - Controlled by `--alpha`

3. **Throughput curvature**
   - Multiply by a quadratic factor across wavenumber axis
   - Controlled by `--beta`

A small jitter factor is applied to these parameters per block (default implementation may jitter by ±20% per block; can be increased in code if desired).

### 7.2 Axis perturbations (blockwise)

Applied by resampling intensity arrays onto a modified axis.

1. **Global axis shift**
   - Shifts all wavenumbers by a random amount
   - Controlled by `--shift-std` (standard deviation in wavenumber units)

2. **Smooth polynomial warp**
   - Creates a smooth, non-linear distortion across the axis
   - Controlled by `--warp-strength`

Resampling uses cubic interpolation.

### 7.3 Notes on realism vs. difficulty

- Increasing `--shift-std` and `--warp-strength` is usually the most effective way to reduce PLS performance because it breaks alignment of peaks.
- Increasing `--fwhm` reduces fine structure.
- Increasing `--alpha`/`--beta` changes slope/curvature and can simulate throughput variations.

---

## 8) Running the generator

### 8.1 Generate plastic, SS, or both

From `Final_Synthesis/Generator`:

```bash
python main.py --generate --target plastic --with-labels
python main.py --generate --target ss --with-labels
python main.py --generate --target both --with-labels
```

### 8.2 Disable perturbations (debug)

```bash
python main.py --generate --target plastic --no-perturb --with-labels
```

### 8.3 Tune perturbation strength (example)

These settings produced a dataset where PLS achieved approximately **R² ≈ 0.85**:

```bash
python main.py --generate --target plastic --with-labels \
  --fwhm 10 --alpha 0.10 --beta 0.05 --shift-std 2.5 --warp-strength 0.003
```

### 8.4 Compression choice

The generator supports HDF5 compression choices:

- `--compression lzf` (fast; good default)
- `--compression gzip` (smaller; slower)
- `--compression none` (fastest; largest files)

---

## 9) Output format: `.h5` structure

Each generated dataset is an HDF5 file, typically:

- `outputs/synthetic_plastic.h5`
- `outputs/synthetic_ss.h5`

### 9.1 Datasets

Inside the H5:

- `/X` — float32, shape `(N, 1024)`
  - Each row is one synthetic spectrum
- `/y_mab` — float32, shape `(N,)`
  - mAb concentration (g/L) for each row
  - Derived from the mAb filename concentration parsed during caching
- `/wn` — float32, shape `(1024,)`
  - Shared wavenumber axis

Optional:

- `/indices` — int32, shape `(N, 4)` when `--with-labels` is used  
  Columns: `(d, a, b, c)` referencing turbidity, BSA, mAb, MP indices used to generate that row.

### 9.2 Attributes

The generator also stores attributes like:
- `nA_BSA`, `nB_mAb`, `nC_MP`, `nD_turb`
- `seed`
- `apply_perturb`

(Recommended: also store perturbation knob values as attributes.)

---

## 10) Inspecting the generated dataset (`inspect_h5.py`)

The inspection script:

- Prints shapes/dtypes/ranges
- Checks for NaNs and Infs in a random sample
- Plots random spectra
- Plots histogram of `y_mab`

### 10.1 Why sorted indices matter

`h5py` requires fancy-index arrays to be in increasing order; the inspection script therefore:
- sorts indices before sampling slices
- reads random plots row-by-row (small `k`) to avoid fancy indexing errors

### 10.2 Command

```bash
python inspect_h5.py --file outputs/synthetic_plastic.h5
python inspect_h5.py --file outputs/synthetic_ss.h5
```

Optional args include:
- `--n-spectra`
- `--seed`
- `--hist-sample`
- `--check-sample`

---

## 11) Training a PLS model (`train_pls.py`)

The PLS script trains a model to predict `y_mab` from `X`.

### 11.1 Efficient sampling strategy

To keep IO fast and avoid `h5py` fancy-index constraints, the script samples **contiguous blocks** of rows for train and test. This gives:
- fast reads
- reproducible sampling
- no index-order errors

### 11.2 Command

```bash
python train_pls.py --file outputs/synthetic_plastic.h5 --components 20
```

Common tuning parameters:
- `--components` (PLS latent variables)
- `--n-train`, `--n-test`
- `--block-len` (contiguous block size)
- `--seed`

### 11.3 Interpreting results

- If R² is too high (too easy / too linear), increase perturbations:
  - `--shift-std`, `--warp-strength` (most effective)
  - `--fwhm`, `--alpha`, `--beta` (secondary)
- If R² is too low (too distorted), reduce those knobs.

---

## 12) How to change / extend the generator

### 12.1 Changing perturbation “knobs”

Use CLI args:
- `--fwhm`
- `--alpha`
- `--beta`
- `--shift-std`
- `--warp-strength`

These feed directly into the block perturbation functions.

### 12.2 Changing per-block randomness

If you want stronger variability without adding new perturbation types, increase jitter inside `apply_peakshape_block(...)`.  
(Example: change from `0.2` to `0.6` or `1.0`.)

### 12.3 Adding new perturbations

Recommended pattern:
- operate on `tmp` (shape `(nB,nC,1024)`) or `flat` (shape `(nB*nC,1024)`)
- keep operations vectorized across rows
- avoid per-spectrum Python loops unless necessary

Common additions:
- additive noise / shot noise
- baseline drift (polynomial)
- random scaling per component
- per-spectrum turbidity scaling

---

## 13) Troubleshooting

### 13.1 Missing columns in CSV

If a CSV does not contain `Wavenumber` or `Intensities`, caching will fail.  
Fix the turbidity CSV column names to match or make the loader more flexible.

### 13.2 Cache missing

If `--generate` says cache files are missing:
- run `python main.py --build-cache` first
- verify `Generator/cache/` contains the expected `.npy` files

### 13.3 Disk usage

H5 files can be large. With float32, size is approximately:

\[
\text{bytes} \approx N \times 1024 \times 4
\]

Compression reduces size at the cost of generation and read speed.

---

## 14) Quick “how-to” summary

1) Build cache once:
```bash
python main.py --build-cache
```

2) Generate dataset:
```bash
python main.py --generate --target plastic --with-labels \
  --fwhm 10 --alpha 0.10 --beta 0.05 --shift-std 2.5 --warp-strength 0.003
```

3) Inspect:
```bash
python inspect_h5.py --file outputs/synthetic_plastic.h5
```

4) Train PLS:
```bash
python train_pls.py --file outputs/synthetic_plastic.h5 --components 20
```

---

## 15) Final outputs

- `outputs/synthetic_plastic.h5`
- `outputs/synthetic_ss.h5`

Each contains:
- `X` (spectra)
- `y_mab` (mAb concentration label)
- `wn` (axis)
- optional `indices` (component indices)

These files are ready for downstream ML training (PLS, SVR, CNNs, etc.) using HDF5 streaming.
