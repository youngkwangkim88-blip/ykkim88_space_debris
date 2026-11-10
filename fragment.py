"""Expected calibrated SNR and SNR-10 ranges from surface, spin and a moving PSF."""

import argparse
import csv
from dataclasses import dataclass
import numpy as np
from scipy.constants import astronomical_unit, c, h
import sky
from diffraction import kernel
from fragment_math import attitudes, factor, render_fft, threshold, variance_coefficients

PAPER_SPEED = 939.56668616  # arcseconds per second
NATIVE_PIXELS = 3428**2


@dataclass
class FragmentInputs:
    material_fits: dict
    photon_irradiance: float
    psf: np.ndarray
    initial_attitudes: np.ndarray
    spin_axes: np.ndarray
    focal_m: float = sky.FOCAL_M
    aperture_m: float = sky.APERTURE_M


def setup(band):
    with (sky.DATA / "optics/material_laws.csv").open(encoding="utf-8") as stream:
        fits = [
            {"material": row["material"], **{k: float(row[k]) for k in ("rho_d", "rho_s", "m")}}
            for row in csv.DictReader(stream)
            if row["band"] == band
        ]
    solar, optics, qe = sky.instrument(band)
    weight = sky.WAVE * solar * optics * qe
    materials = {fit["material"]: fit for fit in fits}
    materials["grey"] = {"rho_d": 0.175, "rho_s": 0.0, "m": 50.0}
    with np.load(sky.DATA / "optics/spin_states.npz") as states:
        return FragmentInputs(
            materials,
            np.trapezoid(weight * 1e-9 / (h * c), sky.WAVE),
            kernel(sky.WAVE, weight),
            states["q0"],
            states["axes"],
        )


def ranges(band, shape, material, exposure, period, speed, illumination, *, snr_distance=329.5):
    if not np.isfinite(speed) or speed < 0 or not np.isfinite(snr_distance) or snr_distance <= 0:
        raise ValueError("Speed must be finite and nonnegative and distance finite and positive")
    p = setup(band)
    scenes = sky.sightlines()
    light = scenes[illumination]
    times, q = attitudes(p, shape, period)
    keep = times < exposure
    flux = factor(p, q[:, keep], shape, material, light["target_phase_far_sun_deg"])
    flux *= (astronomical_unit / 1e3 / light["sun_distance_km"]) ** 2
    image = render_fft(p, flux, times[keep], exposure, speed)
    moments = [np.sum(image**k, axis=1) for k in range(1, 5)]
    for name, scene in scenes.items():
        rate = sum(sky.background(scene, band).values())
        distance, scale = threshold(moments, 100.0, exposure, rate, NATIVE_PIXELS)
        snr = calibrated_snr(moments, exposure, rate, (100.0 / snr_distance) ** 2)
        peak = image.max(axis=1) * scale + (rate + 0.7754788379076724) * exposure
        if np.any(peak >= 7400):
            raise ValueError("Threshold solution exceeds the modeled pixel full well")
        if np.any(
            image.max(axis=1) * (100.0 / snr_distance) ** 2 + (rate + 0.7754788379076724) * exposure >= 7400
        ):
            raise ValueError("At the requested SNR distance at least one state exceeds pixel full well")
        yield name, np.quantile(distance, [0.05, 0.5, 0.95]), np.median(snr)


def calibrated_snr(moments, exposure, rate, scale):
    s2 = moments[1]
    c0, c1, c2 = variance_coefficients(moments, exposure, rate, NATIVE_PIXELS)
    return scale * s2 / np.sqrt(c0 + scale * c1 + scale**2 * c2)


def run(
    band="E12",
    shape="plate",
    material="grey",
    exposure=4.0,
    period=10.0,
    speed=PAPER_SPEED,
    illumination="weak",
    snr_distance=329.5,
):
    results = list(
        ranges(band, shape, material, exposure, period, speed, illumination, snr_distance=snr_distance)
    )
    reference_cases = (
        {"weak": "weak_date", "strong": "strong_fixed", "strong_lunar": "lunar_fixed"}
        if illumination == "weak"
        else {"strong": "strong_date", "strong_lunar": "strong_date_lunar"}
    )
    if period == 10 and speed == PAPER_SPEED:
        with (sky.DATA / "ranges_reference.csv").open(encoding="utf-8") as stream:
            reference = {
                r["case"]: r
                for r in csv.DictReader(stream)
                if (r["band"], r["shape"], r["material"], float(r["T"])) == (band, shape, material, exposure)
            }
        for name, quantiles, _ in results:
            if name in reference_cases:
                row = reference[reference_cases[name]]
                np.testing.assert_allclose(
                    quantiles, [float(row[k]) for k in ("p05", "p50", "p95")], atol=0.005, rtol=0
                )
    for name, (p05, p50, p95), snr in results:
        print(
            f"{name:14} R10 P05 / P50 / P95 = {p05:.1f} / {p50:.1f} / {p95:.1f} km; median SNR at {snr_distance:g} km = {snr:.3f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--band", choices=("E5", "E12"), default="E12")
    parser.add_argument("--shape", choices=("plate", "cuboid"), default="plate")
    parser.add_argument("--material", choices=("grey", "alum", "mli", "solar_cell", "spback"), default="grey")
    parser.add_argument("--exposure", type=float, choices=(0.5, 4.0), default=4.0)
    parser.add_argument("--period", choices=("fixed", "1", "10", "100"), default="10")
    parser.add_argument("--speed", type=float, default=PAPER_SPEED, help="apparent angular speed, arcsec/s")
    parser.add_argument(
        "--illumination",
        choices=("weak", "strong"),
        default="weak",
        help="August or July target illumination",
    )
    parser.add_argument("--snr-distance", type=float, default=329.5, help="range in km for the SNR column")
    args = parser.parse_args()
    if args.period != "fixed":
        args.period = float(args.period)
    run(**vars(args))
