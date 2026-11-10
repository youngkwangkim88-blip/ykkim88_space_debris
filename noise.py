"""Weighted variance for a fixed template with fresh shot and read noise."""

import numpy as np


def weak_streak(inputs, settings):
    pars = settings["weak_streak"]
    image = inputs["weak_streak_e"]
    weights = image / image.sum()
    sky = pars["sky_rate"] * pars["exposure_s"]
    dark = pars["dark_rate"] * pars["exposure_s"]
    read = pars["read_e"]
    rng = np.random.Generator(np.random.PCG64(pars["seed"]))
    estimates = []
    for _ in range(pars["draws"]):
        observed = rng.poisson(image + sky + dark) + rng.normal(0, read, image.shape)
        estimates.append(np.sum(weights * (observed - sky - dark)))
    prediction = np.sum(weights**2 * (image + sky + dark + read**2))
    return "weak_streak_pre_ADC", pars["draws"], np.var(estimates, ddof=1), prediction
