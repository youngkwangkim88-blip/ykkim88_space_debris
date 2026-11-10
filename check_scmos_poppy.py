"""Cross-check the virtual sCMOS pixel and trail PSFs with POPPY.

Install the pinned requirements plus poppy==1.1.2 and run
    python check_scmos_poppy.py

POPPY propagates the same ideal circular pupil and paraxial defocus using
512 samples across the pupil and 32 x 32 image midpoints per detector pixel.
The paper side calls the published diffraction kernel and moving-image renderer
at their native discretizations. Both images retain the light lost outside
the crop. The reported difference is the largest absolute pixel residual in
the window, divided by the peak of the paper image in that same window.
"""

from types import SimpleNamespace

import numpy as np
import poppy

from diffraction import kernel
from fragment_math import render_fft
from sky import APERTURE_M, FOCAL_M, WAVE, instrument


SUBPIXELS = 32
WIDTH = 31
PITCH_M = 2.8e-6


def trail_windows(image, reference):
    """Integrate a steady horizontal trail starting at (+0.25, +0.25) pixel."""
    exposure = 0.5
    speed_arcsec_s = 130.44231045
    times = (np.arange(1024) + 0.5) * exposure / 1024
    shifts = times * speed_arcsec_s / (206265 * PITCH_M / FOCAL_M)
    x = np.arange(-6, 11)
    y = np.arange(-6, 7)

    # Sum each detector row at POPPY's sampling. Interpolation of the cumulative
    # row sums handles fractional overlaps at both ends of each moving pixel.
    left = (WIDTH // 2 + x - 0.25 - shifts[:, None]) * SUBPIXELS
    edges = np.arange(image.shape[1] + 1)
    independent = []
    for row in y:
        bottom = int((WIDTH // 2 + row - 0.25) * SUBPIXELS)
        strip = image[bottom:bottom + SUBPIXELS].sum(axis=0)
        summed = np.r_[0.0, strip.cumsum()]
        signal = np.interp(left + SUBPIXELS, edges, summed) - np.interp(left, edges, summed)
        independent.append(signal.mean(axis=0))

    state = SimpleNamespace(psf=reference, focal_m=FOCAL_M)
    rates = np.full((1, len(times)), 1 / exposure)
    paper = render_fft(state, rates, times, exposure, speed_arcsec_s)
    # render_fft returns a flattened 65-row stripe with the origin at (32, 32).
    return np.asarray(independent), paper.reshape(65, -1)[np.ix_(32 + y, 32 + x)]


def residual_percent(independent, paper):
    return 100 * np.max(np.abs(independent - paper)) / np.max(paper)


def main():
    plate_scale = np.rad2deg(PITCH_M / FOCAL_M) * 3600
    radius = APERTURE_M / 2
    f_number = FOCAL_M / APERTURE_M
    z4 = 25e-6 / (16 * np.sqrt(3) * f_number**2)
    system = poppy.OpticalSystem(npix=512, pupil_diameter=APERTURE_M)
    system.add_pupil(poppy.CircularAperture(radius=radius))
    system.add_pupil(poppy.ZernikeWFE(radius=radius, coefficients=[0, 0, 0, z4]))
    system.add_detector(pixelscale=plate_scale, fov_pixels=WIDTH, oversample=SUBPIXELS)
    print(f"POPPY {poppy.__version__}, scale {plate_scale:.7f} arcsec/pixel, Noll Z4 {z4*1e9:.4f} nm RMS")

    # Independent reader's values in percentage points, rounded to four decimals.
    published = {"E5": (0.5492, 0.1818), "E12": (0.6808, 0.1978)}
    for band, expected in published.items():
        solar, transmission, qe = instrument(band)
        weights = WAVE * solar * transmission * qe
        edges = np.searchsorted(weights.cumsum(), weights.sum() * np.array([1/3, 2/3]))
        bins = np.split(np.arange(WAVE.size), edges)
        wavelengths = np.array([np.average(WAVE[group], weights=weights[group]) for group in bins])
        masses = np.array([weights[group].sum() for group in bins])
        source = {"wavelengths": wavelengths * 1e-9, "weights": masses / masses.sum()}
        image = system.calc_psf(source=source, normalize="first")[0].data
        static = image.reshape(WIDTH, SUBPIXELS, WIDTH, SUBPIXELS).sum(axis=(1, 3))

        reference = kernel(WAVE, weights)
        # The paper's kernel is already pixel-integrated; centres are 8 samples apart.
        indices = reference.shape[0] // 2 + 8 * np.arange(-(WIDTH // 2), WIDTH // 2 + 1)
        paper_static = reference[np.ix_(indices, indices)]
        static_error = residual_percent(static, paper_static)
        trail_error = residual_percent(*trail_windows(image, reference))
        print(f"{band} static (31x31): peak POPPY/paper {static.max():.8f}/{paper_static.max():.8f}, "
              f"window flux {static.sum():.8f}/{paper_static.sum():.8f}, residual {static_error:.5f}%")
        print(f"{band} trail (13x17, 0.5 s, 130.4423 arcsec/s): residual {trail_error:.5f}%")
        # One unit of the last reported decimal checks numerical reproduction.
        # The tolerance measures neither lens uncertainty nor SNR-distance error.
        np.testing.assert_allclose((static_error, trail_error), expected, rtol=0, atol=0.0001)


if __name__ == "__main__":
    main()
