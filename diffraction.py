"""Scalar diffraction and detector pixel integration used for the paper PSF."""

import numpy as np
from scipy.special import roots_legendre, j0
from scipy.ndimage import convolve1d

FOCAL_MM = 36.66929888837269


def profiles(camera, offsets_um, nodes, dr=0.005, rmax=24.0, nquad=512, exact=False):
    """Absolute PSF density per pixel squared; never normalize away crop loss."""
    rho, weight = roots_legendre(nquad)
    rho, weight = (rho + 1) / 2, weight / 2
    radial = np.arange(0, rmax + dr / 2, dr)
    offsets = np.asarray(offsets_um, float)
    p, n = camera["pitch_um"], camera["f_number"]
    opd = offsets[None, :] / (8 * n * n) * (rho[:, None] ** 2 - 0.5)
    scales = np.ones(len(offsets))
    if exact:
        f = FOCAL_MM * 1000
        pupil_r = f / (2 * n) * rho[:, None]
        z = f + offsets[None, :]
        # Rationalized spherical paths; uniform scalar pupil, not a lens prescription.
        opd = pupil_r**2 / (np.sqrt(f * f + pupil_r**2) + f) - pupil_r**2 / (np.sqrt(z * z + pupil_r**2) + z)
        scales = 1 + offsets / f
    result = np.zeros((len(radial), len(offsets)))
    for wavelength_nm, spectral_weight in nodes:
        wavelength_um = wavelength_nm / 1000
        phase = np.exp(2j * np.pi * opd / wavelength_um)
        if exact:
            for col, scale in enumerate(scales):
                q = np.pi * p * radial / (wavelength_um * n * scale)
                amp = 2 * (j0(q[:, None] * rho[None, :]) @ (weight * rho * phase[:, col]))
                result[:, col] += (
                    spectral_weight * np.pi / 4 * (p / (wavelength_um * n * scale)) ** 2 * abs(amp) ** 2
                )
        else:
            q = np.pi * p * radial / (wavelength_um * n)
            amp = 2 * (j0(q[:, None] * rho[None, :]) @ ((weight * rho)[:, None] * phase))
            result += spectral_weight * np.pi / 4 * (p / (wavelength_um * n)) ** 2 * abs(amp) ** 2
    return radial, result


def kernel(wavelength, weight):
    boundaries = np.searchsorted(np.cumsum(weight), weight.sum() * np.array([1 / 3, 2 / 3]))
    cells = np.split(np.arange(len(weight)), boundaries)
    nodes = np.array([[np.average(wavelength[g], weights=weight[g]), weight[g].sum()] for g in cells])
    nodes[:, 1] /= nodes[:, 1].sum()
    radius, density = profiles(
        dict(pitch_um=2.8, f_number=2.0), [25.0], nodes, dr=0.025, rmax=20 * 5.5 / 2.8, nquad=128
    )
    x = np.arange(-256, 257) / 8
    values = np.interp(np.hypot(x[:, None], x[None, :]), radius, density[:, 0], right=0)
    box = np.ones(9) / 8
    box[[0, -1]] *= 0.5
    return convolve1d(convolve1d(values, box, axis=0, mode="constant"), box, axis=1, mode="constant")
