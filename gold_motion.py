"""Monte Carlo golden for the exposure-averaged position of a moving light source.

This fixed example uses a circular chief orbit, Hill-Clohessy-Wiltshire (HCW)
relative motion, a fixed camera and uniform illumination. The only random input
is the initial relative orbital state. Detector noise and estimation from pixels
are outside this experiment. The image-plane coordinates are tangent-plane
ratios scaled to arcseconds.

Ported from AI_SBSSA_v2 at dabc7e54ae72ef154c8be6a1383b1c59b4b7e5f3:
wi/WI-OPNN-RESEARCH-PLAN/work/tests/test_model_f_lcov.py
(test_hcw_exposure_covariance_uses_one_shared_state_and_matches_monte_carlo),
work/scripts/model_f/f_hcw_exposure.py, f_hcw_covariance.py and f_illumination.py
in the same capsule, and wi/WI-PHYSICS-ASTRODYNAMICS-MOTION-MODELS/work/hcw.py.
"""

import math

import numpy as np


ARCSEC_PER_RAD = 180 * 3600 / math.pi
MEAN_MOTION_RAD_S = 0.001
# Initial radial, along-track and cross-track position [m], then velocity [m/s].
INITIAL_STATE = np.array([1200.0, -600.0, 100.0, 0.4, -0.3, 0.2])
STATE_SIGMA = np.array([1.0, 1.0, 10.0, 0.01, 0.01, 0.03])
OBSERVER_M = np.array([0.0, 0.0, -10000.0])
EXPOSURE_S = (30.0, 90.0)


def _position_transition(t):
    """First three rows of the HCW state-transition matrix at epoch t."""
    n = MEAN_MOTION_RAD_S
    s, c = math.sin(n * t), math.cos(n * t)
    return np.array([
        [4 - 3*c, 0, 0, s/n, 2*(1-c)/n, 0],
        [6*(s-n*t), 1, 0, -2*(1-c)/n, (4*s-3*n*t)/n, 0],
        [0, 0, c, 0, 0, s/n],
    ])


def _nodes(order):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    start, end = EXPOSURE_S
    return (start + end)/2 + (end - start)*nodes/2, weights/2


def _prediction(order):
    times, weights = _nodes(order)
    positions = []
    mean_jacobian = np.zeros((2, 6))
    for time, weight in zip(times, weights):
        transition = _position_transition(time)
        x, y, z = transition @ INITIAL_STATE - OBSERVER_M
        positions.append(ARCSEC_PER_RAD * np.array([x, y]) / z)
        jacobian = ARCSEC_PER_RAD * np.array([[1/z, 0, -x/z**2], [0, 1/z, -y/z**2]])
        # Every photon time depends on the same initial orbital error.
        mean_jacobian += weight * (jacobian @ transition)

    angular_path = np.asarray(positions)
    centroid = weights @ angular_path
    deviations = angular_path - centroid
    path_spread = (deviations.T * weights) @ deviations
    centroid_covariance = mean_jacobian @ np.diag(STATE_SIGMA**2) @ mean_jacobian.T
    return centroid, centroid_covariance, path_spread


def run():
    """Check the historical seed, quadrature convergence, centroid offset and spread."""
    draws, seed = 16000, 719
    centroid, predicted, spread = _prediction(16)
    _, finer, _ = _prediction(32)
    np.testing.assert_allclose(predicted, finer, rtol=1e-10)

    rng = np.random.default_rng(seed)
    states = INITIAL_STATE + rng.normal(size=(draws, 6)) * STATE_SIGMA
    measured_centroids = np.zeros((draws, 2))
    times, weights = _nodes(16)
    for time, weight in zip(times, weights):
        # Project each perturbed trajectory through the full nonlinear camera
        # equation, then integrate it. No analytic Jacobian enters this path.
        rays = states @ _position_transition(time).T - OBSERVER_M
        measured_centroids += weight * ARCSEC_PER_RAD * rays[:, :2] / rays[:, 2, None]
    empirical = np.cov(measured_centroids.T, ddof=1)
    np.testing.assert_allclose(empirical, predicted, rtol=0.04, atol=0.3)

    midpoint_ray = _position_transition(sum(EXPOSURE_S) / 2) @ INITIAL_STATE - OBSERVER_M
    midpoint = ARCSEC_PER_RAD * midpoint_ray[:2] / midpoint_ray[2]
    offset = float(np.linalg.norm(centroid - midpoint))
    spread_ratio = float(np.trace(spread) / np.trace(predicted))
    if offset <= 0.5 or spread_ratio <= 10:
        raise AssertionError("Historical curvature and photon-spread controls failed")
    return {
        "draws": draws,
        "seed": seed,
        "exposure_bounds_s": list(EXPOSURE_S),
        "predicted_centroid_covariance_arcsec2": predicted.tolist(),
        "sample_centroid_covariance_arcsec2": empirical.tolist(),
        "photon_path_spread_arcsec2": spread.tolist(),
        "centroid_offset_from_mid_exposure_arcsec": offset,
        "path_spread_to_centroid_covariance_trace_ratio": spread_ratio,
    }
