"""Rebuild the raw DN and calibrated electrons of Figure 3.2(a), first 4 s."""

import argparse
from pathlib import Path

import numpy as np


CASE = Path(__file__).resolve().parent / "data/figure32/e12_weak_frame1.npz"
EXPOSURE_S = 4.0
GAIN_E_PER_DN = 10.0 / 3.0
READ_NOISE_E = 3.3
DARK_E_PER_PIXEL_S = 0.7754788379076724
FLAT_E = 5000.0


def read_detector(rng, expected_e, bias_e):
    """Shot noise, independent Gaussian read noise, fixed bias, and a 12-bit ADC."""
    noisy_e = rng.poisson(expected_e) + rng.normal(0, READ_NOISE_E, expected_e.shape) + bias_e
    return np.clip(np.rint(noisy_e / GAIN_E_PER_DN), 0, 4095).astype(np.uint16)


def detector_and_masters(stripe_shape):
    """Build the fixed pixel maps and the duration-matched masters once per sequence."""
    rng = np.random.default_rng(8202)
    sensitivity = rng.normal(1.0, 0.01, stripe_shape)
    log_sigma = np.sqrt(np.log(1.0 + 0.32**2))
    dark_rate = DARK_E_PER_PIXEL_S * np.exp(rng.normal(-log_sigma**2 / 2, log_sigma, stripe_shape))
    bias_e = rng.normal(128.0 * GAIN_E_PER_DN, 14.6, stripe_shape)

    def make_master(frame_count, lamp_e):
        expected_e = EXPOSURE_S * dark_rate + sensitivity * lamp_e
        total_e = np.zeros(stripe_shape)
        for _ in range(frame_count):
            total_e += read_detector(rng, expected_e, bias_e) * GAIN_E_PER_DN
        return total_e / frame_count

    master_dark_e = make_master(64, 0.0)
    flat_minus_dark_e = make_master(32, FLAT_E) - master_dark_e
    return sensitivity, dark_rate, bias_e, master_dark_e, flat_minus_dark_e


def reproduce(output=None):
    with np.load(CASE, allow_pickle=False) as paper:
        # The archived noiseless source is float32. Use double precision for the
        # detector calculation, matching the original generator's arithmetic.
        target_e = paper["target_e"].astype(np.float64)
        height, width = target_e.shape
        offsets = paper["crop_x"]
        stripe_shape = (height, int(offsets[-1]) + width)
        crop = np.s_[:, int(offsets[0]):int(offsets[0]) + width]
        sensitivity, dark_rate, bias_e, master_dark_e, flat_minus_dark_e = detector_and_masters(stripe_shape)
        expected = sensitivity[crop] * (target_e + float(paper["sky_rate"]) * EXPOSURE_S)
        expected += dark_rate[crop] * EXPOSURE_S
        raw_dn = read_detector(np.random.default_rng(88331), expected, bias_e[crop])
        science_minus_dark_e = raw_dn * GAIN_E_PER_DN - master_dark_e[crop]
        calibrated = science_minus_dark_e / (flat_minus_dark_e[crop] / FLAT_E)
        calibrated_e = calibrated.astype(np.float32)
        np.testing.assert_array_equal(raw_dn, paper["raw_dn"])
        np.testing.assert_array_equal(calibrated_e, paper["calibrated_e"])

    # For comparison only, substitute the realized stripe mean for the
    # nominal lamp. This stripe samples a fraction of the full detector.
    stripe_mean_e = flat_minus_dark_e.mean()
    stripe_calibrated = science_minus_dark_e / (flat_minus_dark_e[crop] / stripe_mean_e)
    max_change_e = np.max(np.abs(stripe_calibrated - calibrated))
    max_saved_difference_e = np.max(np.abs(stripe_calibrated.astype(np.float32) - calibrated_e))
    print(f"Figure 3.2(a): all {raw_dn.size} raw DN and float32 calibrated pixels match.")
    print(f"Master stripe {height} x {stripe_shape[1]}: mean(flat - dark) = {stripe_mean_e:.7f} e-/pixel")
    print(f"Using the stripe mean instead of 5000 e-/pixel: maximum change in float64 = {max_change_e:.9f} e-")
    print(f"Maximum change after both images are saved as float32: {max_saved_difference_e:.9f} e-")
    if output is not None:
        with Path(output).open("wb") as stream:
            np.savez_compressed(stream, raw_dn=raw_dn, calibrated_e=calibrated_e,
                                master_dark_e=master_dark_e, flat_minus_dark_e=flat_minus_dark_e)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, help="write the reconstructed frame and full-stripe masters as NPZ")
    reproduce(cli.parse_args().output)
