"""
PeakShapeProcessor -- variable broadening + throughput tilt / curve.

Broadening:
    Convolve spectrum with a Gaussian kernel of given FWHM.

Throughput tilt:
    Multiply spectrum by  f(nu) = 1 + alpha*(nu - nu_center)

Throughput curve:
    Multiply spectrum by  f(nu) = 1 + beta*(nu - nu_center)^2
"""

import numpy as np
from typing import Optional


class PeakShapeProcessor:
    """
    Apply instrument-response effects to a spectrum.

    Parameters
    ----------
    broadening_fwhm : float
        Full-width at half-maximum (cm-1) for Gaussian broadening.
        Set to 0 to skip broadening.
    tilt_alpha : float
        Linear throughput tilt coefficient.
    curve_beta : float
        Quadratic throughput curvature coefficient.
    """

    def __init__(
        self,
        broadening_fwhm: float = 0.0,
        tilt_alpha: float = 0.0,
        curve_beta: float = 0.0,
    ):
        self.broadening_fwhm = broadening_fwhm
        self.tilt_alpha = tilt_alpha
        self.curve_beta = curve_beta

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def apply(
        self,
        spectrum: np.ndarray,
        wavenumbers: np.ndarray,
        rng: Optional[np.random.Generator] = None,
        broadening_fwhm: Optional[float] = None,
        tilt_alpha: Optional[float] = None,
        curve_beta: Optional[float] = None,
    ) -> np.ndarray:
        """
        Apply peak-shape and throughput effects.

        If parameters are *None* the instance defaults are used.
        If an ``rng`` is provided, small random jitter is added to
        broadening and tilt to introduce per-sample variation.
        """
        rng = rng or np.random.default_rng()

        fwhm  = broadening_fwhm if broadening_fwhm is not None else self.broadening_fwhm
        alpha = tilt_alpha if tilt_alpha is not None else self.tilt_alpha
        beta  = curve_beta if curve_beta is not None else self.curve_beta

        # Per-sample jitter (±20 % relative)
        fwhm  = max(0.0, fwhm  * (1 + 0.2 * rng.standard_normal()))
        alpha = alpha * (1 + 0.2 * rng.standard_normal())
        beta  = beta  * (1 + 0.2 * rng.standard_normal())

        result = spectrum.copy()

        # 1. Gaussian broadening
        if fwhm > 0:
            result = self._broaden(result, wavenumbers, fwhm)

        # 2. Throughput tilt + curve
        result = self._throughput(result, wavenumbers, alpha, beta)

        return result

    # ------------------------------------------------------------------
    # Broadening
    # ------------------------------------------------------------------
    @staticmethod
    def _broaden(
        spectrum: np.ndarray,
        wavenumbers: np.ndarray,
        fwhm: float,
    ) -> np.ndarray:
        """Convolve with Gaussian kernel of given FWHM (cm-1)."""
        dnu = np.abs(np.mean(np.diff(wavenumbers)))  # grid spacing
        sigma = fwhm / 2.3548  # FWHM -> sigma
        n_pts = int(np.ceil(4 * sigma / dnu))  # 4-sigma truncation
        if n_pts < 1:
            return spectrum

        x = np.arange(-n_pts, n_pts + 1) * dnu
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        kernel /= kernel.sum()

        # Mode 'same' preserves length; 'reflect' avoids edge artefacts
        padded = np.pad(spectrum, n_pts, mode='reflect')
        convolved = np.convolve(padded, kernel, mode='same')
        return convolved[n_pts:-n_pts]

    # ------------------------------------------------------------------
    # Throughput
    # ------------------------------------------------------------------
    @staticmethod
    def _throughput(
        spectrum: np.ndarray,
        wavenumbers: np.ndarray,
        alpha: float,
        beta: float,
    ) -> np.ndarray:
        """
        Multiply by smooth throughput function:
            f(nu) = 1 + alpha*(nu - nu_c) + beta*(nu - nu_c)^2

        Normalised so that f(nu_c) == 1.
        """
        nu_c = 0.5 * (wavenumbers.min() + wavenumbers.max())
        nu_range = wavenumbers.max() - wavenumbers.min()
        nu_norm = (wavenumbers - nu_c) / nu_range  # in [-0.5, 0.5]

        f = 1.0 + alpha * nu_norm + beta * nu_norm ** 2
        return spectrum * f
