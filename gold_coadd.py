"""Moving-aperture flux covariance with one dark and flat shared by all reads.

A new numerical check of the A1 coadd variance, independent of the historical
astrometric covariance experiment. Both use the term covariance for different
observables. Here each coordinate is a fitted flux, in calibrated electrons.
"""

import numpy as np
from coadd import DARK, L, ND, NF, R2, master_variances, variance_parts


def run():
    duration = 0.5
    yy, xx = np.indices((7, 15))
    profiles = np.array([np.exp(-((xx - x)**2 + (yy - 3)**2) / 2) for x in (4, 5, 11)])
    profiles = profiles.reshape(3, -1)
    profiles /= profiles.sum(axis=1, keepdims=True)
    weights = profiles / (profiles**2).sum(axis=1, keepdims=True)
    optical = 180 * profiles + 18  # optical electrons per pixel in each exposure
    dark = DARK * duration
    science_var = optical + dark + R2
    vd, vu = master_variances(duration)
    pixels = xx.size

    def variance(coefficients):
        return variance_parts(coefficients, optical, science_var, 1, vd, vu, pixels)

    # Polarization of the published coadd quadratic form gives Cov(F_i,F_j).
    basis = np.eye(3)[:, :, None] * weights
    diagonal = np.array([variance(row) for row in basis])
    parts = np.array([[(variance(left + right) - diagonal[i] - diagonal[j]) / 2
                       for j, right in enumerate(basis)] for i, left in enumerate(basis)])
    predicted_cal = parts[:, :, 1:].sum(axis=2)
    predicted_total = parts.sum(axis=2)

    rng = np.random.default_rng(20261110)
    calibration, total = [], []
    for _ in range(60):
        size = (500, pixels)
        # Exact Poisson draws for the summed exposures of each master. The
        # combined additive read/ADC term uses its Gaussian-equivalent variance.
        dark_error = rng.poisson(ND * dark, size) / ND - dark + rng.normal(0, np.sqrt(R2 / ND), size)
        flat_error = rng.poisson(NF * (L + dark), size) / NF - (L + dark) + rng.normal(0, np.sqrt(R2 / NF), size)
        flat = L + flat_error - dark_error
        flat /= flat.mean(axis=1, keepdims=True)
        science_error = (rng.poisson(optical + dark, (size[0], *optical.shape)) - (optical + dark)
                         + rng.normal(0, np.sqrt(R2), (size[0], *optical.shape)))
        reduced = (optical - dark_error[:, None, :]) / flat[:, None, :]
        calibration.extend(np.sum((reduced - optical) * weights, axis=2))
        total.extend(np.sum((reduced + science_error / flat[:, None, :] - optical) * weights, axis=2))

    def compare(samples, predicted):
        centered = np.asarray(samples) - np.mean(samples, axis=0)
        products = centered[:, :, None] * centered[:, None, :]
        observed = np.cov(centered.T)
        standard_error = products.std(axis=0, ddof=1) / np.sqrt(len(centered))
        z = np.abs(observed - predicted) / standard_error
        if z.max() > 5:
            raise AssertionError(f"Moving-flux covariance outside five Monte Carlo standard errors: {z}")
        return observed, float(z.max())

    observed_cal, cal_z = compare(calibration, predicted_cal)
    _, total_z = compare(total, predicted_total)
    if not predicted_cal[0, 1] > 10 * abs(predicted_cal[0, 2]):
        raise AssertionError("Overlapping and separated measurement footprints should give distinct covariance")
    return dict(draws=len(total), seed=20261110, coordinates="flux per read, e-",
                predicted_master_covariance_e2=predicted_cal.tolist(),
                measured_master_covariance_e2=observed_cal.tolist(),
                predicted_total_variance_e2=predicted_total.diagonal().tolist(),
                largest_calibration_mc_se=cal_z, largest_total_mc_se=total_z)
