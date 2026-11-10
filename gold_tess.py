"""Replay the historical 21 x 21 TESS floor comparison at its measured epochs."""

import csv
import json
from collections import Counter
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import GCRS, GeocentricMeanEcliptic, SkyCoord, get_sun
from astropy.time import Time
from scipy.interpolate import RegularGridInterpolator

import tess
from skycore import sky_background as sky
from tess_operators import sun_directions, unit
from tess_floor import bundled_example


HERE = Path(__file__).resolve().parent
K_VEGA = 1.021408688828167
PLANE_BAND = (0.9194799697121285, 1.2060717511072077)
LADDER_RMS_BUDGET = 0.03664009515291639


def interpolation_check(delta, beta, magnitude):
    """Read the ETC axes in published orientation and interpolate flux with SciPy."""
    with (tess.HERE / "zodiacal_mu.csv").open(encoding="utf-8") as stream:
        reader = csv.reader(line for line in stream if line.strip() and not line.startswith("#"))
        latitudes = np.array(next(reader)[1:], dtype=float)
        rows = list(reader)
    longitudes = np.array([row[0] for row in rows], dtype=float)
    grid = np.array([[float(v) if v != "SE" else np.nan for v in row[1:]] for row in rows])
    i, j = np.where(np.isfinite(grid))
    if len(i) != 184:
        raise ValueError("Expected 184 published zodiacal knots")
    actual, status = tess.TABLE.lookup(longitudes[i], latitudes[j])
    np.testing.assert_equal(status, sky.STATUS_ZL_OK)
    np.testing.assert_allclose(actual, grid[i, j], atol=1e-9, rtol=0)

    axes = (longitudes, latitudes)
    points = np.column_stack((delta, abs(beta)))
    unsupported_weight = RegularGridInterpolator(axes, np.isnan(grid).astype(float))(points)
    if np.any(unsupported_weight > 1e-12):
        raise ValueError("A historic field uses an unpublished near-Sun knot")
    flux = np.nan_to_num(10 ** (-0.4 * grid))
    expected = -2.5 * np.log10(RegularGridInterpolator(axes, flux)(points))
    np.testing.assert_allclose(magnitude, expected, atol=1e-9, rtol=0)
    return len(i)


def independent_geometry(ra, dec, btjd, delta, beta, elongation):
    """Check the model against Astropy frames and the low-accuracy Meeus solar longitude."""
    time = Time(btjd + 2457000.0, format="jd", scale="tdb")
    fields = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame=GCRS(obstime=time))
    sun = get_sun(time)
    ecliptic = GeocentricMeanEcliptic(equinox="J2000", obstime=time)
    field_e = fields.transform_to(ecliptic)
    sun_e = sun.transform_to(ecliptic)
    folded = abs((field_e.lon.deg - sun_e.lon.deg + 180) % 360 - 180)
    for actual, expected in ((delta, folded), (beta, field_e.lat.deg), (elongation, fields.separation(sun).deg)):
        np.testing.assert_allclose(actual, expected, atol=0.02, rtol=0)

    # Meeus, Astronomical Algorithms, ch. 25. Remove general precession to J2000.
    t = (time.jd - 2451545.0) / 36525.0
    mean_lon = 280.46646 + 36000.76983*t + 0.0003032*t**2
    anomaly = np.radians(357.52911 + 35999.05029*t - 0.0001537*t**2)
    centre = ((1.914602 - 0.004817*t - 0.000014*t**2)*np.sin(anomaly)
              + (0.019993 - 0.000101*t)*np.sin(2*anomaly) + 0.000289*np.sin(3*anomaly))
    sun_lon = mean_lon + centre - 0.00569 - (5029.0966*t + 1.11113*t**2)/3600
    a, d, obliquity = np.radians(ra), np.radians(dec), np.radians(84381.406/3600)
    ray_lon = np.degrees(np.arctan2(np.sin(a)*np.cos(d)*np.cos(obliquity) + np.sin(d)*np.sin(obliquity),
                                    np.cos(a)*np.cos(d)))
    simple_beta = np.degrees(np.arcsin(np.sin(d)*np.cos(obliquity) - np.cos(d)*np.sin(obliquity)*np.sin(a)))
    np.testing.assert_allclose(delta, abs((ray_lon - sun_lon + 180) % 360 - 180), atol=0.02, rtol=0)
    np.testing.assert_allclose(beta, simple_beta, atol=0.02, rtol=0)
    return folded, field_e.lat.deg


def run():
    """Raise on a changed computation and return the two historical observation diagnostics."""
    with (HERE / "data/goldens/tess/cutouts.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 59 or len({(r["position"], r["sector"]) for r in rows}) != 59:
        raise ValueError("Expected 59 distinct TESS cutouts")

    def column(key):
        return np.array([float(row[key]) for row in rows])

    ra, dec, btjd = column("ra_deg"), column("dec_deg"), column("t_floor_btjd")
    rays, suns = unit(ra, dec), sun_directions(btjd)
    angles = np.array([sky.helioecliptic(ray, sun)[:2] for ray, sun in zip(rays, suns)])
    delta, beta = np.degrees(angles).T
    elongation = np.degrees(sky.angular_separation(rays, suns))
    magnitude, status = tess.TABLE.lookup(delta, abs(beta))
    np.testing.assert_equal(status, sky.STATUS_ZL_OK)
    knot_count = interpolation_check(delta, beta, magnitude)
    oracle_delta, oracle_beta = independent_geometry(ra, dec, btjd, delta, beta, elongation)
    oracle_magnitude, oracle_status = tess.TABLE.lookup(oracle_delta, abs(oracle_beta))
    np.testing.assert_equal(oracle_status, sky.STATUS_ZL_OK)
    np.testing.assert_allclose(magnitude, oracle_magnitude, atol=0.0012337459065321355, rtol=0)
    np.testing.assert_allclose(beta, column("ecl_lat_deg_nominal"), atol=0.01, rtol=0)
    group = np.select(
        [column("field_angle_deg") > 14, abs(beta) <= 3, (delta <= 30) & (abs(beta) >= 50)],
        ["excluded_corner_glow", "ecliptic_plane", "sun_side_high_latitude"], default="anti_solar_survey",
    )
    np.testing.assert_array_equal(group, [row["group_at_freeze"] for row in rows])

    density = np.array([tess.zodiacal(ray, sun, tess.COLOUR) for ray, sun in zip(rays, suns)])
    rates = density * K_VEGA * column("pixel_solid_angle_sr")
    reference = column("reference_zodiacal_e_per_px_s")
    observed = column("measured_floor_e_per_px_s")
    modeled = rates + column("minor_mid_e_per_px_s")
    if not np.all(np.isfinite([rates, modeled, observed])) or np.any(rates <= 0) or np.any(observed <= 0):
        raise ValueError("Each cutout requires a positive finite rate and measured floor")
    np.testing.assert_allclose(rates, reference, atol=0, rtol=1e-9)
    plane = group == "ecliptic_plane"
    ladder = group == "anti_solar_survey"
    if plane.sum() != 5 or ladder.sum() != 48:
        raise ValueError("The plane or anti-solar population changed")
    plane_ratio = float(np.median(observed[plane] / modeled[plane]))
    area = column("pixel_solid_angle_arcsec2")[ladder]
    x, y = modeled[ladder]/area, observed[ladder]/area
    slope, intercept = np.linalg.lstsq(np.column_stack((x, np.ones_like(x))), y, rcond=None)[0]
    rms = float(np.sqrt(np.mean(((y - (slope*x + intercept))/y)**2)))
    np.testing.assert_allclose([plane_ratio, rms], [0.9182976965626416, 0.048107993004095496], atol=0, rtol=1e-9)
    if plane_ratio >= PLANE_BAND[0] or rms <= LADDER_RMS_BUDGET:
        raise ValueError("Historical observation-gap classification changed")
    return {
        "cutouts": len(rows),
        "sector3_from_pixels": bundled_example(),
        "finite_table_knots": knot_count,
        "cutouts_by_group": dict(Counter(row["group_at_freeze"] for row in rows)),
        "max_relative_rate_replay_error": float(np.max(abs(rates/reference - 1))),
        "plane_observed_over_model_median": plane_ratio,
        "plane_two_sigma_band": list(PLANE_BAND),
        "plane_observational_status": "OPEN",
        "anti_solar_fractional_rms_after_affine_fit": rms,
        "anti_solar_rms_budget": LADDER_RMS_BUDGET,
        "anti_solar_observational_status": "OPEN",
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
