"""Check the bright-background 4 s virtual-camera variance from the public v30 inputs."""
import argparse
import json
from pathlib import Path

import numpy as np


def check(release):
    base = release / "data" / "goldens" / "photometry"
    cfg = json.loads((base / "cameras.json").read_text(encoding="utf-8"))
    detector = cfg["detectors"]["virtual_sCMOS"]
    with np.load(base / "native_trails.npz", allow_pickle=False) as arrays:
        profile = arrays["virtual_sCMOS_P4"]
        sky = arrays["virtual_sCMOS_sky_rate"]
    n = 16384
    seed = 20261111
    flux = cfg["flux_e"]
    offset = (sky + detector["dark"]) * cfg["exposure_s"]
    mean = flux * profile + offset
    weights = profile / np.sum(profile**2)
    half_step = 5 / 3
    per_pixel = mean + detector["read"]**2 + half_step**2 / 3
    variance = np.sum(weights**2 * per_pixel)
    # Independent fourth cumulants add; the Gaussian term has zero fourth cumulant.
    cumulant = np.sum(weights**4 * (mean - 2 * half_step**4 / 15))
    def relative_sd_error(draws):
        return 0.5 * np.sqrt(2 / (draws - 1) + cumulant / (draws * variance**2))
    rng = np.random.Generator(np.random.PCG64(seed))
    samples = []
    for _ in range(n // 32):
        shape = (32, *mean.shape)
        x = rng.poisson(mean, size=shape) + rng.normal(0, detector["read"], size=shape)
        x += rng.uniform(-half_step, half_step, size=shape)
        samples.extend(np.sum((x - offset) * weights, axis=(1, 2)))
    print(json.dumps(dict(draws=n, seed=seed, analytic_sd=float(np.sqrt(variance)),
                          measured_sd=float(np.std(samples, ddof=1)),
                          old_relative_se=float(relative_sd_error(1024)),
                          new_relative_se=float(relative_sd_error(n)),
                          mean_flux=float(np.mean(samples)), source=float(np.sum(weights*flux*profile)))))
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path, nargs="?", default=Path(__file__).resolve().parents[2])
    check(parser.parse_args().release)
