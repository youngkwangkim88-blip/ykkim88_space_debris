"""Compare scene-linear Cycles pixel integrals with directional optical cross sections."""

import argparse
from collections import Counter
import json
from math import fsum, isclose, log10, pi, sqrt
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from fragment_math import factor


DATA = Path(__file__).resolve().parent / "data" / "goldens" / "blender"
RELATIVE_TOLERANCE = 0.002


def _cross_section(case, archive, rerender_dir):
    key = case["id"]
    if rerender_dir is None:
        red = archive[key]
    else:
        red = np.load(Path(rerender_dir) / (key + ".npy"), allow_pickle=False)[:, :, 0]
    resolution = case["resolution"]
    if red.shape != (resolution, resolution) or not np.isfinite(red).all():
        raise ValueError(f"{key}: invalid linear radiance image")
    return float(red.sum(dtype=np.float64)) * (case["ortho_scale_m"] / resolution) ** 2


def _check_relative(key, measured, expected):
    error = abs(measured / expected - 1)
    if not error < RELATIVE_TOLERANCE:
        raise AssertionError(f"{key}: {measured:.12g} versus {expected:.12g} ({error:.3%})")
    return error


def _public_plate(case):
    # Unit aperture area and unit incident photon irradiance leave C / range^2.
    rho = case["rho"]
    specular_fraction = case["f"] if case["law"] == "declared_blinn" else 0
    material = {"rho_d": rho * (1 - specular_fraction), "rho_s": rho * specular_fraction, "m": case["m"]}
    state = SimpleNamespace(photon_irradiance=1, aperture_m=2 / sqrt(pi), material_fits={"fixture": material})
    q = np.asarray([case["quaternion_wxyz"]])
    phase = np.degrees(np.arccos(np.dot(case["sun"], case["view"])))
    # The fixture and the public 10 cm plate have identical aspect ratios.
    linear_scale = case["dimensions_m"][0] / 0.1
    np.testing.assert_allclose(case["dimensions_m"], np.array([0.1, 0.1, 0.0125]) * linear_scale)
    return float(factor(state, q, "plate", "fixture", phase)[0]) * 100_000**2 * linear_scale**2


def run(*, poses_from=None, exposure_from=None):
    """Replay the retained pixels, or substitute fresh RGB .npy renders for either group."""
    fixture = json.loads((DATA / "cases.json").read_text(encoding="utf-8"))
    poses = fixture["poses"]
    exposure = fixture["exposure"]
    nodes = exposure["cases"]
    if Counter(n["component"] for n in nodes) != {"diffuse": 9, "specular": 27}:
        raise ValueError("Incorrect exposure quadrature population")
    if min(n["frame_weight"] for n in nodes) <= 0 or not isclose(fsum(n["frame_weight"] for n in nodes), 1):
        raise ValueError("Invalid exposure weights")

    static_errors, live_errors, node_errors = [], [], []
    native_flux_terms, cycles_flux_terms = [], []
    plane_area = None
    with np.load(DATA / "cycles_linear_red.npz", allow_pickle=False) as archive:
        for case in poses:
            measured = _cross_section(case, archive, poses_from)
            expected = case["expected_cross_section_m2_sr"]
            error = _check_relative(case["id"], measured, expected)
            if case["id"] == "unit_plane":
                plane_area = measured
            else:
                static_errors.append(error)
            if case["shape"] == "plate":
                public = _public_plate(case)
                np.testing.assert_allclose(public, expected, rtol=1e-10, atol=0, err_msg=case["id"])
                live_errors.append(_check_relative(case["id"], measured, public))
        for case in nodes:
            measured = _cross_section(case, archive, exposure_from)
            expected = case["expected_cross_section_m2_sr"]
            node_errors.append(_check_relative(case["id"], measured, expected))
            scale = case["frame_weight"] / case["range_m"]**2
            native_flux_terms.append(scale * expected)
            cycles_flux_terms.append(scale * measured)
    if plane_area is None or len(static_errors) != 18 or len(live_errors) != 6:
        raise ValueError("Incomplete static pose references")

    sun_mv = -3.51 + 2.5 * log10(0.2 / (4 * 10_000**2))
    native_mv = sun_mv - 2.5 * log10(fsum(native_flux_terms))
    cycles_mv = sun_mv - 2.5 * log10(fsum(cycles_flux_terms))
    cached_mv = exposure["cached_exposure_mv"]
    if abs(native_mv - cached_mv) > 1e-10 or abs(cycles_mv - cached_mv) > -2.5 * log10(1 - RELATIVE_TOLERANCE):
        raise AssertionError(f"Exposure magnitude: native {native_mv}, Blender {cycles_mv}, cached {cached_mv}")
    return {
        "unit_plane_area_m2": plane_area,
        "matched_pose_laws": len(static_errors),
        "max_pose_relative_error_pct": 100 * max(static_errors),
        "live_public_plate_cases": len(live_errors),
        "max_live_plate_relative_error_pct": 100 * max(live_errors),
        "exposure_nodes": len(node_errors),
        "max_node_relative_error_pct": 100 * max(node_errors),
        "native_exposure_mv": native_mv,
        "cached_exposure_mv": cached_mv,
        "blender_exposure_mv": cycles_mv,
        "blender_minus_cache_mag": cycles_mv - cached_mv,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poses-from", type=Path, help="Directory of freshly rendered static RGB planes")
    parser.add_argument("--exposure-from", type=Path, help="Directory of freshly rendered exposure nodes")
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))
