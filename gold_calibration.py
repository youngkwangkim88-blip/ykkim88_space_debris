"""A1 finite dark/flat master Monte Carlo, originally calibration_check.py (v27).

The target is a stationary spot on a nonuniform detector. A fresh pair of
masters is drawn per trial, then a separate noisy science observation.
"""

import numpy as np


def run():
    seed = 2026110508
    rng = np.random.default_rng(seed)
    pixels = 32**2
    dark_count, flat_count, lamp = 64, 32, 5000.
    read, adc_step = 13., 10/3
    sensitivity = 1 + rng.normal(0, 0.01, pixels)
    sensitivity /= sensitivity.mean()
    log_sigma = np.sqrt(np.log1p(0.32**2))
    dark = 2.992 * 4 * rng.lognormal(-log_sigma**2/2, log_sigma, pixels)
    yy, xx = np.indices((32, 32))
    source = (50 * np.exp(-((xx - 15.5)**2 + (yy - 15.5)**2) / 4)).ravel()
    optical = source + 35
    weights = source / source.sum()

    read_var = read**2 + adc_step**2 / 12
    vd = (dark + read_var) / dark_count
    vf = (sensitivity * lamp + dark + read_var) / flat_count
    a = weights / sensitivity
    k = weights * optical / (sensitivity * lamp) - np.sum(weights * optical) / (pixels * lamp)
    science_var = np.sum(weights**2 * (sensitivity * optical + dark + read_var) / sensitivity**2)
    reuse = -2 * np.sum(a * k * vd)
    master_var = np.sum(a**2 * vd) + np.sum(k**2 * (vf + vd)) + reuse

    shape = (500, pixels)
    def master_error(mean, count):
        return (rng.poisson(count * mean, shape) / count - mean
                + rng.normal(0, read / np.sqrt(count), shape)
                + rng.uniform(-adc_step / 2, adc_step / 2, shape) / np.sqrt(count))

    cal, total = [], []
    for _ in range(40):
        dark_error = master_error(dark, dark_count)
        flat_error = master_error(sensitivity * lamp + dark, flat_count)
        flat = sensitivity * lamp + flat_error - dark_error
        flat /= flat.mean(axis=1, keepdims=True)
        ideal = sensitivity * optical - dark_error
        cal.extend((ideal / flat) @ weights)
        mean = sensitivity * optical + dark
        noise = (rng.poisson(mean, shape) - mean + rng.normal(0, read, shape)
                 + rng.uniform(-adc_step / 2, adc_step / 2, shape))
        total.extend(((sensitivity * optical + noise - dark_error) / flat) @ weights)

    ratios = [np.var(cal, ddof=1) / master_var, np.var(total, ddof=1) / (science_var + master_var)]
    # Frozen supplement outputs check seed and procedure; the scientific
    # acceptance of the historical run was ±4% around a variance ratio of 1.
    np.testing.assert_allclose(ratios, [1.0016368549498524, 0.9905159083899793], rtol=1e-8)
    np.testing.assert_allclose(ratios, [1, 1], atol=0.04, rtol=0)
    np.testing.assert_allclose(reuse, -0.0030446281600906245, rtol=1e-10)
    return {"seed": seed, "draws": len(cal), "master_variance_ratio": float(ratios[0]),
            "total_variance_ratio": float(ratios[1]), "dark_reuse_covariance_term_e2": float(reuse)}
