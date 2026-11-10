"""Face photometry, attitude samples and moving-PSF integration. Rates refer to 100 km."""

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.fft import rfft, irfft, next_fast_len
from rotation import propagate_torque_free_batch, _body_axes_batch


def factor(state, q, shape, material, phase):
    """Rate in electrons/s at 100 km and 1 AU for illuminated, visible faces."""
    axes = np.stack(_body_axes_batch(q), axis=-2)
    a = np.deg2rad(phase)
    sun = np.array([np.sin(a), 0, np.cos(a)])
    view = np.array([0.0, 0.0, 1.0])
    half = (sun + view) / np.linalg.norm(sun + view)
    total = np.zeros(q.shape[:-1])
    thick = 0.0125 if shape == "plate" else 0.1
    f = state.material_fits[material]
    m = f["m"]
    K = (m + 2) * (m + 4) / (8 * (m + 2 ** (-m / 2)))
    for axis, sign, area in [
        (2, 1, 0.01),
        (2, -1, 0.01),
        (0, 1, 0.1 * thick),
        (0, -1, 0.1 * thick),
        (1, 1, 0.1 * thick),
        (1, -1, 0.1 * thick),
    ]:
        normal = sign * axes[..., axis, :]
        total += (
            area
            * np.maximum(normal @ sun, 0)
            * np.maximum(normal @ view, 0)
            * (f["rho_d"] + f["rho_s"] * K * np.maximum(normal @ half, 0) ** m)
        )
    return total * state.photon_irradiance * np.pi * (state.aperture_m / 2) ** 2 / (np.pi * 100e3**2)


def attitudes(state, shape, period, refine=1):
    """Integrate free motion from shutter open on the retained 4 s midpoint grid."""
    n_states = len(state.initial_attitudes)
    count = (4096 if period == 1 else 2048) * refine
    t = (np.arange(count) + 0.5) * 4 / count
    if period == "fixed":
        return t, np.broadcast_to(state.initial_attitudes[:, None, :], (n_states, count, 4))
    dims = np.array([0.1, 0.1, 0.0125 if shape == "plate" else 0.1])
    inertia = (
        np.array([dims[1] ** 2 + dims[2] ** 2, dims[0] ** 2 + dims[2] ** 2, dims[0] ** 2 + dims[1] ** 2]) / 12
    )
    q, _ = propagate_torque_free_batch(
        state.initial_attitudes,
        state.spin_axes * 2 * np.pi / period,
        np.broadcast_to(inertia, (n_states, 3)),
        t,
        np.full(n_states, period / (768 * refine)),
        t0_s=0.0,
    )
    return t, q


def render_fft(state, rates, t, T, angular):
    """Accumulate each light curve along the same trail using the pixel-integrated PSF."""
    os, half = 8, 32
    plate = 206265 * 2.8e-6 / state.focal_m
    speed = angular / plate
    length = speed * T
    nx = int(np.ceil(length)) + 2 * half + 1
    ny = 2 * half + 1
    pos = t * speed * os
    lo = np.floor(pos).astype(int)
    frac = pos - lo
    nc = max(int(np.ceil(length * os)) + 2, 2)
    coeff = (
        np.array(
            [
                np.bincount(lo, weights=r * (1 - frac), minlength=nc)
                + np.bincount(lo + 1, weights=r * frac, minlength=nc)
                for r in rates
            ]
        )
        * T
        / len(t)
    )
    x = -half - 0.25 + np.arange(ny * os) / os
    y = np.arange(-half, half + 1) - 0.25
    gy, gx = np.meshgrid(y, x, indexing="ij")
    h = map_coordinates(
        state.psf, np.array([(gy + half) * os, (gx + half) * os]), order=1, mode="constant", cval=0.0
    )
    nfft = next_fast_len(h.shape[1] + nc - 1)
    hf = rfft(h, n=nfft, axis=-1, workers=1)
    result = []
    for start in range(0, len(coeff), 8):
        cf = rfft(coeff[start : start + 8], n=nfft, axis=-1, workers=1)
        conv = irfft(cf[:, None, :] * hf[None, :, :], n=nfft, axis=-1, workers=1)
        result.append(conv[:, :, : nx * os : os].reshape(-1, ny * nx))
    return np.maximum(np.concatenate(result), 0.0)


def variance_coefficients(moments, t, rate, pixels, dark=0.7754788379076724):
    """Coefficients of weighted variance in source amplitude for finite dark and flat masters."""
    s1, s2, s3, s4 = moments
    b = rate * t
    v0 = dark * t + 3.3**2 + (10 / 3) ** 2 / 12
    vd = v0 / 64
    flat = 5000.0
    vu = (flat + v0) / 32 + vd
    c0 = s2 * (b + v0 + vd) + (vu * b * b / flat**2 - 2 * vd * b / flat) * (s2 - s1 * s1 / pixels)
    c1 = s3 + (2 * vu * b / flat**2 - 2 * vd / flat) * (s3 - s1 * s2 / pixels)
    c2 = vu / flat**2 * (s4 - s2 * s2 / pixels)
    return c0, c1, c2


def threshold(moments, distance, t, rate, pixels, dark=0.7754788379076724):
    """Return the SNR-10 range given image moments at the reference distance."""
    s2 = moments[1]
    c0, c1, c2 = variance_coefficients(moments, t, rate, pixels, dark)
    A = s2 * s2 - 100 * c2
    assert np.all(A > 0)
    amplitude = (100 * c1 + np.sqrt((100 * c1) ** 2 + 400 * A * c0)) / (2 * A)
    assert np.allclose(amplitude * s2 / np.sqrt(c0 + amplitude * c1 + amplitude**2 * c2), 10)
    return distance / np.sqrt(amplitude), amplitude
