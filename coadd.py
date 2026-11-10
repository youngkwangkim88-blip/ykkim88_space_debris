"""Single 4 s exposure versus sums of duration-matched short reads at fixed total time."""

import argparse
import csv
import numpy as np
from scipy.constants import astronomical_unit
from fragment import setup, PAPER_SPEED as OMEGA, NATIVE_PIXELS as NPIX
from fragment_math import attitudes, factor, render_fft
import sky

DARK = 0.7754788379076724
R2 = 3.3**2 + (10 / 3) ** 2 / 12
DISTANCE = 329.5232806914059
L, ND, NF = 5000.0, 64, 32


def master_variances(t):
    """One duration-matched pair shared across all science frames in a scheme."""
    vd = (DARK * t + R2) / ND
    vf = (L + DARK * t + R2) / NF
    return vd, vf + vd


def variance_parts(w, u, v, a, vd, vu, pixels=NPIX):
    """Variance of sum_jp w_jp y_jp, including full-field flat normalization.

    a=1 for original reads and a=N for the single summed observation.
    u is the background-inclusive reduced optical expectation (no dark).
    Cov(Dhat, Fhat-Dhat)=-Var(Dhat) is retained with its negative sign.
    """
    aw = np.sum(w * a, axis=0)
    uw = np.sum(w * u, axis=0)
    return np.array(
        [
            np.sum(w * w * v),
            vd * np.sum(aw * aw),
            vu / L**2 * (np.sum(uw * uw) - np.sum(uw) ** 2 / pixels),
            -2 * vd / L * (np.sum(aw * uw) - np.sum(aw) * np.sum(uw) / pixels),
        ]
    )


def inverse_covariance(s, u, v, a, vd, vu, pixels=NPIX):
    """Exact inverse of the first-order uniform-calibration covariance.

    Each pixel has diagonal read/photon variance plus a two-dimensional
    dark/flat update across time; normalized flat adds one global rank-two
    update. Thus no full (time*pixels)^2 matrix is materialized.
    """
    a = np.broadcast_to(a, s.shape)
    b = np.array([[vd, -vd / L], [-vd / L, vu / L**2]])
    bi = np.linalg.inv(b)
    va, uu = a / v, u / v
    m00 = np.sum(a * va, axis=0) + bi[0, 0]
    m01 = np.sum(a * uu, axis=0) + bi[0, 1]
    m11 = np.sum(u * uu, axis=0) + bi[1, 1]
    det = m00 * m11 - m01 * m01

    def local(x):
        z0, z1 = np.sum(va * x, axis=0), np.sum(uu * x, axis=0)
        c0, c1 = (m11 * z0 - m01 * z1) / det, (m00 * z1 - m01 * z0) / det
        return x / v - va * c0 - uu * c1

    ks, ka, ku = local(s), local(a), local(u)
    z = np.array([np.sum(a * ks), np.sum(u * ks)])
    m = np.array([[np.sum(a * ka), np.sum(a * ku)], [np.sum(u * ka), np.sum(u * ku)]])
    g = np.array([[0.0, vd / (pixels * L)], [vd / (pixels * L), -vu / (pixels * L**2)]])
    correction = np.linalg.solve(np.eye(2) + g @ m, g @ z)
    return ks - ka * correction[0] - ku * correction[1]


def statistics(s, b, t):
    n = len(s)
    vd, vu = master_variances(t)
    u = s + b * t
    v = s + (b + DARK) * t + R2
    total = s.sum(axis=0, keepdims=True)
    us = u.sum(axis=0, keepdims=True)
    vs = v.sum(axis=0, keepdims=True)
    parts = variance_parts(total, us, vs, n, vd, vu)
    numerator = np.sum(total * total)
    fixed = numerator / np.sqrt(parts.sum())
    wi = inverse_covariance(total, us, vs, n, vd, vu)
    summed_optimal = np.sqrt(np.sum(total * wi))
    wt = inverse_covariance(s, u, v, 1.0, vd, vu)
    retained = np.sqrt(np.sum(s * wt))
    # An exactly recoverable deterministic sum cannot add information.
    assert retained + 1e-9 >= summed_optimal >= fixed - 1e-9
    # An independent contraction verifies the inverse-covariance result.
    check = np.sum(wt * s) / np.sqrt(variance_parts(wt, u, v, 1.0, vd, vu).sum())
    assert np.isclose(check, retained, rtol=2e-12)
    return dict(
        source_e=float(total.sum()),
        peak_e=float(total.max()),
        snr_sum_fixed=float(fixed),
        snr_sum_optimal=float(summed_optimal),
        snr_frames_optimal=float(retained),
        snr_time_loss_pct=float(100 * (1 - summed_optimal / retained)),
        **dict(
            zip(
                ["var_science", "var_dark_master", "var_flat_master", "var_dark_flat_cross"],
                map(float, parts),
            )
        ),
        science_white_read_variance_e2=n * R2,
        dark_master_variance_per_pixel_e2=n * n * vd,
        flat_relative_variance=vu / L**2,
    )


def split_templates(state, flux, t, dt, n, exposure):
    """Partition v27 midpoint integration cells, preserving their actual duration.

    Short-read boundaries need not coincide with the 2048-cell quadrature grid.
    An overlap integral apportions a cell without restarting the body's phase
    or translating the target back to the start of the detector trail.
    """
    lo, hi = np.arange(n) * exposure, (np.arange(n) + 1) * exposure
    weights = np.maximum(0.0, np.minimum(hi[:, None], t + dt / 2) - np.maximum(lo[:, None], t - dt / 2))
    assert np.allclose(weights.sum(axis=1), exposure, atol=2e-14)
    speed = OMEGA / (206265 * 2.8e-6 / state.focal_m)
    tpad = 4 + 1 / speed
    weighted_rates = flux * weights * len(t) / tpad
    return render_fft(state, weighted_rates, t, tpad, OMEGA) * (100 / DISTANCE) ** 2


def run(shape="plate", band="E12", scene="weak"):
    state = setup(band)
    state.initial_attitudes, state.spin_axes = state.initial_attitudes[:1], state.spin_axes[:1]
    t, q = attitudes(state, shape, 10.0)
    scenes = sky.sightlines()
    light = scenes["weak"]
    flux = factor(state, q, shape, "grey", light["target_phase_far_sun_deg"])
    flux *= (astronomical_unit / 1000 / light["sun_distance_km"]) ** 2
    rate = sum(sky.background(scenes[scene], band).values())
    case = {"weak": "weak_date", "strong": "strong_fixed", "strong_lunar": "lunar_fixed"}[scene]
    filename = "pseudo4_comparison.csv" if shape == "plate" else "pseudo4_cuboid.csv"
    with (sky.DATA / filename).open(encoding="utf-8") as f:
        references = {r["scheme"]: r for r in csv.DictReader(f) if r["band"] == band and r["case"] == case}
    native_e = None
    fields = ("snr_sum_fixed", "snr_sum_optimal", "snr_frames_optimal")
    for scheme, reads in (("native_4s", 1), ("coadd_2x2s", 2), ("coadd_8x0p5s", 8), ("coadd_40x0p1s", 40)):
        dt = 4 / len(t)
        stamps = split_templates(state, flux, t, dt, reads, 4 / reads)
        stats = statistics(stamps, rate, 4 / reads)
        if native_e is None:
            native_e = stats["source_e"]
        np.testing.assert_allclose(stats["source_e"], native_e, rtol=1e-10)
        snrs = [stats[k] for k in fields]
        np.testing.assert_allclose(snrs, [float(references[scheme][k]) for k in fields], rtol=5e-7)
        print(
            f"{reads:2} x {4/reads:g} s: fixed summed SNR {snrs[0]:.3f}, optimal summed {snrs[1]:.3f}, optimal time-resolved {snrs[2]:.3f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", choices=("plate", "cuboid"), default="plate")
    parser.add_argument("--band", choices=("E5", "E12"), default="E12")
    parser.add_argument("--scene", choices=("weak", "strong", "strong_lunar"), default="weak")
    run(**vars(parser.parse_args()))
