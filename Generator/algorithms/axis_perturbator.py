"""
AxisPerturbator -- global peak shift + smooth nonlinear warping.

Global shift:
    y'(nu) = y(nu + delta_nu)

Smooth warping:
    nu' = nu + delta(nu)
    delta(nu) is a low-order polynomial or spline function.
"""

import numpy as np
from scipy.interpolate import interp1d
from typing import Optional


class AxisPerturbator:
    """
    Apply axis perturbations (shift + warp) to a spectrum.

    Parameters
    ----------
    shift_std : float
        Standard deviation (cm-1) of the random global shift.
    warp_strength : float
        Magnitude control for polynomial warp coefficients.
    """

    def __init__(
        self,
        shift_std: float = 1.0,
        warp_strength: float = 0.001,
    ):
        self.shift_std = shift_std
        self.warp_strength = warp_strength

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def perturb(
        self,
        spectrum: np.ndarray,
        wavenumbers: np.ndarray,
        rng: Optional[np.random.Generator] = None,
        apply_shift: bool = True,
        apply_warp: bool = True,
    ) -> np.ndarray:
        """
        Apply shift and/or warp to ``spectrum``.

        Parameters
        ----------
        spectrum : np.ndarray   (n_wl,)
        wavenumbers : np.ndarray (n_wl,)
        rng : numpy Generator, optional
        apply_shift, apply_warp : bool

        Returns
        -------
        np.ndarray  -- perturbed spectrum on the *original* grid.
        """
        rng = rng or np.random.default_rng()
        result = spectrum.copy()
        wl = wavenumbers.copy()

        if apply_shift:
            result = self._global_shift(result, wl, rng)

        if apply_warp:
            result = self._polynomial_warp(result, wl, rng)

        return result

    # ------------------------------------------------------------------
    # Shift
    # ------------------------------------------------------------------
    def _global_shift(
        self,
        spectrum: np.ndarray,
        wavenumbers: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Shift spectrum by delta_nu drawn from N(0, shift_std)."""
        delta = rng.normal(0, self.shift_std)

        interp_func = interp1d(
            wavenumbers,
            spectrum,
            kind='cubic',
            bounds_error=False,
            fill_value=(spectrum[0], spectrum[-1]),
        )
        shifted_wl = wavenumbers + delta
        return interp_func(shifted_wl)

    # ------------------------------------------------------------------
    # Polynomial warp
    # ------------------------------------------------------------------
    def _polynomial_warp(
        self,
        spectrum: np.ndarray,
        wavenumbers: np.ndarray,
        rng: np.random.Generator,
        order: int = 3,
    ) -> np.ndarray:
        """
        Apply a smooth polynomial warping to the axis.

            nu' = nu + c2*(nu-nu_bar)^2 + c3*(nu-nu_bar)^3 + ...

        Coefficients are drawn from N(0, warp_strength).
        """
        nu_bar = np.mean(wavenumbers)
        nu_range = wavenumbers.max() - wavenumbers.min()
        nu_norm = (wavenumbers - nu_bar) / nu_range  # in [-0.5, 0.5]

        # Draw small polynomial coefficients (skip constant and linear)
        coeffs = rng.normal(0, self.warp_strength, order + 1)
        coeffs[0] = 0.0  # no constant offset
        coeffs[1] = 0.0  # no linear shift (handled by global shift)

        delta_nu = np.polyval(coeffs[::-1], nu_norm) * nu_range

        warped_wl = wavenumbers + delta_nu

        interp_func = interp1d(
            wavenumbers,
            spectrum,
            kind='cubic',
            bounds_error=False,
            fill_value=(spectrum[0], spectrum[-1]),
        )
        return interp_func(warped_wl)

    # ------------------------------------------------------------------
    # Spline warp (optional alternative)
    # ------------------------------------------------------------------
    def spline_warp(
        self,
        spectrum: np.ndarray,
        wavenumbers: np.ndarray,
        rng: np.random.Generator,
        n_knots: int = 5,
    ) -> np.ndarray:
        """
        Smooth spline warping with random knot displacements.
        Endpoints are pinned (zero displacement).
        """
        from scipy.interpolate import CubicSpline

        knot_pos = np.linspace(
            wavenumbers.min(), wavenumbers.max(), n_knots
        )
        displacements = rng.normal(0, self.warp_strength * 10, n_knots)
        displacements[0] = 0.0
        displacements[-1] = 0.0

        spline = CubicSpline(knot_pos, displacements)
        delta_nu = spline(wavenumbers)

        warped_wl = wavenumbers + delta_nu
        interp_func = interp1d(
            wavenumbers,
            spectrum,
            kind='cubic',
            bounds_error=False,
            fill_value=(spectrum[0], spectrum[-1]),
        )
        return interp_func(warped_wl)
