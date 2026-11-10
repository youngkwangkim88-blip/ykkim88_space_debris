"""Forced photometry on a supplied candidate track, A1 Eq. (2).

Original Model E subpixel placement and fixed-weight estimator.
The caller supplies calibrated residuals, a base variance map, a flat and
a candidate template. All noise terms use calibrated electron units.
"""

import math
import numpy as np


def shift_template_bilinear(template: np.ndarray, centre_y: int, centre_x: int,
                            dx: float, dy: float) -> tuple[np.ndarray, int, int]:
    """Translate a pixel-centered template without losing flux at fractional phase.

    The returned center is the integer anchor in the one-pixel padded raster;
    the fractional offset is applied by bilinear deposition. The caller maps
    this raster to image pixels using ``anchor - returned_center``.
    """
    template = np.asarray(template, dtype=np.float64)
    if template.ndim != 2 or not np.isfinite(template).all():
        raise ValueError("template must be a finite raster")
    if not (math.isfinite(dx) and math.isfinite(dy) and -0.5000000001 <= dx <= 0.5000000001
            and -0.5000000001 <= dy <= 0.5000000001):
        raise ValueError("fractional center phase is outside the nearest-pixel interval")
    y0, x0 = math.floor(dy), math.floor(dx)
    fy, fx = dy - y0, dx - x0
    out = np.zeros((template.shape[0] + 2, template.shape[1] + 2), dtype=np.float64)
    for oy, wy in ((y0, 1.0 - fy), (y0 + 1, fy)):
        for ox, wx in ((x0, 1.0 - fx), (x0 + 1, fx)):
            if wy and wx:
                out[1 + oy:1 + oy + template.shape[0],
                    1 + ox:1 + ox + template.shape[1]] += template * (wy * wx)
    return out, int(centre_y + 1), int(centre_x + 1)


def matched_measure(residual: np.ndarray, variance: np.ndarray, flat: np.ndarray,
                    profile: np.ndarray, prediction_xy: tuple[float, float],
                    centre_y: int, centre_x: int) -> dict:
    """Measure signed flux with fixed base-noise weights and source shot variance.

    ``profile`` is normalized to unit total flux on the complete template
    support. ``variance`` is the fixed causal-prefix V0 map. The source term is
    ``sum(w**2 * max(Fhat, 0) * p / flat)`` on usable support pixels.
    """
    residual = np.asarray(residual)
    variance = np.asarray(variance)
    flat = np.asarray(flat)
    profile = np.asarray(profile, dtype=np.float64)
    if residual.ndim != 2 or variance.shape != residual.shape or flat.shape != residual.shape:
        raise ValueError("residual, variance and flat must be same-shaped rasters")
    if profile.ndim != 2 or not np.isfinite(profile).all() or np.any(profile < 0):
        raise ValueError("profile must be a finite non-negative 2-D raster")
    profile_sum = float(profile.sum(dtype=np.float64))
    if not math.isfinite(profile_sum) or profile_sum <= 0:
        raise ValueError("profile has no positive finite mass")
    profile = profile / profile_sum
    px, py = map(float, prediction_xy)
    if not (math.isfinite(px) and math.isfinite(py)):
        return {"status": "INVALID_PREDICTION"}
    # Half-up pixel anchor; retain the fractional phase in the shifted template.
    anchor_x = math.floor(px + 0.5)
    anchor_y = math.floor(py + 0.5)
    dx, dy = px - anchor_x, py - anchor_y

    y0, x0 = int(centre_y), int(centre_x)
    shifted, sy, sx = shift_template_bilinear(profile, y0, x0, dx, dy)
    top, left = anchor_y - sy, anchor_x - sx
    H, W = residual.shape
    iy0, ix0 = max(0, top), max(0, left)
    iy1, ix1 = min(H, top + shifted.shape[0]), min(W, left + shifted.shape[1])
    if iy1 <= iy0 or ix1 <= ix0:
        return {
            "status": "NO_IN_IMAGE_SUPPORT", "prediction_xy": [px, py],
            "support_fraction_in_image": 0.0, "support_fraction_usable": 0.0,
            "edge_clipped": True,
        }
    py0, px0 = iy0 - top, ix0 - left
    patch_p = shifted[py0:py0 + (iy1 - iy0), px0:px0 + (ix1 - ix0)]
    in_image_fraction = float(patch_p.sum(dtype=np.float64))
    r = np.asarray(residual[iy0:iy1, ix0:ix1], dtype=np.float64)
    v = np.asarray(variance[iy0:iy1, ix0:ix1], dtype=np.float64)
    f = np.asarray(flat[iy0:iy1, ix0:ix1], dtype=np.float64)
    valid = np.isfinite(r) & np.isfinite(v) & (v > 0) & np.isfinite(f) & (f > 0)
    usable_p = np.where(valid, patch_p, 0.0)
    usable_fraction = float(usable_p.sum(dtype=np.float64))
    edge_clipped = in_image_fraction < 1.0 - 32.0 * np.finfo(np.float64).eps
    if not np.any(valid) or usable_fraction <= 0:
        return {
            "status": "NO_VALID_SUPPORT", "prediction_xy": [px, py],
            "support_fraction_in_image": in_image_fraction,
            "support_fraction_usable": usable_fraction,
            "edge_clipped": edge_clipped,
        }
    p = usable_p[valid]
    rv, vv, ff = r[valid], v[valid], f[valid]
    denom = float(np.sum((p * p) / vv, dtype=np.float64))
    if not math.isfinite(denom) or denom <= 0:
        return {
            "status": "ZERO_BASE_INFORMATION", "prediction_xy": [px, py],
            "support_fraction_in_image": in_image_fraction,
            "support_fraction_usable": usable_fraction,
            "edge_clipped": edge_clipped,
        }
    weights = (p / vv) / denom
    flux = float(np.sum(weights * rv, dtype=np.float64))
    variance_base = 1.0 / denom
    variance_base_check = float(np.sum(weights * weights * vv, dtype=np.float64))
    variance_source = float(np.sum(weights * weights * max(flux, 0.0) * p / ff,
                                   dtype=np.float64))
    variance_total = variance_base + variance_source
    return {
        "status": "MEASURED", "prediction_xy": [px, py],
        "pixel_anchor_xy": [int(anchor_x), int(anchor_y)],
        "subpixel_phase_xy": [float(dx), float(dy)],
        "flux_e": flux,
        "variance_base_e2": variance_base,
        "variance_base_identity_e2": variance_base_check,
        "variance_source_shot_e2": variance_source,
        "variance_total_e2": variance_total,
        "null_z_base_only": flux / math.sqrt(variance_base),
        "photometric_snr": flux / math.sqrt(variance_total),
        "support_fraction_in_image": in_image_fraction,
        "support_fraction_usable": usable_fraction,
        "valid_support_pixels": int(valid.sum()),
        "profile_support_pixels": int(np.count_nonzero(patch_p)),
        "edge_clipped": edge_clipped,
    }
