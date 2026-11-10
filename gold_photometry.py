"""Historical fixed-template noise experiments tied to the operational estimator."""

import csv
import json
from pathlib import Path

import numpy as np

from noise import native_cameras
from operational_photometry import matched_measure, shift_template_bilinear

DATA = Path(__file__).resolve().parent / "data/goldens/photometry"


def reference_cases():
    # Model E, 17 October 2026. Each row gets the original independent RNG stream.
    profile = np.array([[0.25, 0.75]])
    cases = (
        ("uniform_flat", 80, [1, 1], [4, 4], [True, True]),
        ("nonunit_flat", 60, [0.5, 2], [2, 8], [True, True]),
        ("masked_nonunit_flat", 60, [0.5, 2], [2, 8], [False, True]),
        ("null_source", 0, [1, 1], [4, 4], [True, True]),
        ("independent_eight_frame_median_reference", 0, [1, 1], [4, 4], [True, True]),
    )
    count = 100_000
    for index, (name, flux, flat, variance, mask) in enumerate(cases):
        flat = np.array([flat], dtype=float)
        variance = np.array([variance], dtype=float)
        v0 = np.where(mask, variance, np.inf)

        def measure(residual):
            return matched_measure(residual, v0, flat, profile, (0, 0), 0, 0)

        # The numerator is linear in residuals. Applying the real, masked
        # estimator to two unit impulses gives its weights for all 100k images.
        weights = np.array([measure(row[None, :])["flux_e"] for row in np.eye(2)])
        expected = measure(flux * profile)
        np.testing.assert_allclose(weights @ profile[0], 1)
        np.testing.assert_allclose(expected["flux_e"], flux)
        rng = np.random.default_rng(np.random.SeedSequence([20261017, index]))
        if name == "independent_eight_frame_median_reference":
            reference = np.median(rng.normal(0, 2, size=(count, 8, 2)), axis=1)
            residual = rng.normal(0, 2, size=(count, 2)) - reference
        else:
            residual = rng.poisson(flat * flux * profile, size=(count, 2)) / flat
            residual += rng.normal(size=(count, 2)) * np.sqrt(variance)
        yield name, count, np.var(residual @ weights, ddof=1), expected["variance_total_e2"]


def run():
    with (DATA / "reference.csv").open(encoding="utf-8") as stream:
        references = {row["case"]: row for row in csv.DictReader(stream)}
    results = {}
    def check(trials):
        for case, draws, empirical, predicted in trials:
            row = references[case]
            if draws != int(row["draws_per_case"]):
                raise AssertionError(f"Historical draw count differs for {case}")
            np.testing.assert_allclose(
                [empirical, predicted, empirical / predicted],
                [float(row[key]) for key in ("measured", "predicted", "measured_over_predicted")],
                rtol=1e-8, atol=1e-12, err_msg=case,
            )
            results[case] = dict(draws=draws, quantity=row["quantity"], empirical_to_analytic=float(empirical / predicted))
    check(reference_cases())
    with np.load(DATA / "native_trails.npz", allow_pickle=False) as inputs:
        settings = json.loads((DATA / "cameras.json").read_text(encoding="utf-8"))
        check(native_cameras(inputs, {"native": settings}))
    if results.keys() != references.keys():
        raise AssertionError("Missing a historical experiment")
    profile = np.array([[0.25, 0.75]])
    flat = np.ones_like(profile)
    negative = matched_measure(-12 * profile, 4 * flat, flat, profile, (0, 0), 0, 0)
    np.testing.assert_allclose([negative["flux_e"], negative["variance_source_shot_e2"],
                                negative["photometric_snr"]], [-12, 0, -12 / np.sqrt(6.4)])
    empty = matched_measure(profile, np.full_like(flat, np.inf), flat, profile, (0, 0), 0, 0)
    if empty["status"] != "NO_VALID_SUPPORT":
        raise AssertionError("An empty footprint cannot give a measured SNR")
    shifted, _, _ = shift_template_bilinear(profile, 0, 0, 0.25, -0.375)
    np.testing.assert_allclose(shifted.sum(), 1, atol=1e-15, rtol=0)
    measured = matched_measure(80 * profile, 4 * flat, flat, profile, (0, 0), 0, 0)
    return {"repeated_noise": results, "operational_snr_80e": measured["photometric_snr"],
            "operational_snr_negative": negative["photometric_snr"]}
