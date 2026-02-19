import numpy as np

def apply_peakshape_block(X_block, wn, rng, fwhm=0.0, alpha=0.0, beta=0.0):
    """
    X_block: (..., 1024) float32
    Applies broadening + throughput to the whole block.
    Uses SciPy for fast broadening if available.
    """
    X = X_block  # modify in-place if you want

    # --- Broadening (fast block version) ---
    if fwhm and fwhm > 0:
        from scipy.ndimage import gaussian_filter1d

        dnu = float(np.abs(np.mean(np.diff(wn))))
        sigma = float(fwhm) / 2.3548
        sigma_pts = sigma / dnu

        # reflect edges like your implementation intended
        gaussian_filter1d(X, sigma=sigma_pts, axis=-1, mode="reflect", output=X)

    # --- Throughput tilt/curve (vectorized) ---
    nu_c = 0.5 * (wn.min() + wn.max())
    nu_range = wn.max() - wn.min()
    nu_norm = (wn - nu_c) / nu_range

    f = (1.0 + alpha * nu_norm + beta * (nu_norm ** 2)).astype(np.float32)
    X *= f  # broadcasts over block

    return X


def apply_axis_perturb_block(X_block, wn, rng, shift_std=1.0, warp_strength=0.001, order=3):
    """
    Applies ONE random shift + ONE random warp to the whole block (fast).
    X_block: (..., 1024)
    """
    from scipy.interpolate import interp1d

    X = X_block

    # Global shift
    delta = rng.normal(0.0, shift_std)
    wn_shifted = wn + delta

    f_interp = interp1d(
        wn, X, kind="cubic", axis=-1,
        bounds_error=False,
        fill_value=(X[..., 0], X[..., -1]),
    )
    X = f_interp(wn_shifted).astype(np.float32, copy=False)

    # Polynomial warp
    nu_bar = float(np.mean(wn))
    nu_range = float(wn.max() - wn.min())
    nu_norm = (wn - nu_bar) / nu_range

    coeffs = rng.normal(0.0, warp_strength, order + 1)
    coeffs[0] = 0.0
    coeffs[1] = 0.0
    delta_nu = np.polyval(coeffs[::-1], nu_norm) * nu_range
    wn_warped = wn + delta_nu.astype(np.float32)

    f_interp2 = interp1d(
        wn, X, kind="cubic", axis=-1,
        bounds_error=False,
        fill_value=(X[..., 0], X[..., -1]),
    )
    X = f_interp2(wn_warped).astype(np.float32, copy=False)

    return X
