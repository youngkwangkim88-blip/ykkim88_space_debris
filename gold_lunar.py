"""Reproduce lunar stray-light rates for the historical CMV and v27 sCMOS cases."""

import json
from pathlib import Path

import numpy as np
from scipy.constants import c, h

import sky
from skycore import moon_halo as moon

DATA = Path(__file__).resolve().parent / "data/goldens/lunar"
PST_RECORD = sky.SKY / "liu2021_cubesat_deployable_baffle_pst_expfit_v1.json"


def figure_checks(table, pst):
    """Compare the declared ROLO spectrum and the fitted baffle to published plot readings."""
    def read_curve(name):
        with (DATA / name).open(encoding="ascii") as stream:
            return np.genfromtxt((line for line in stream if not line.startswith("#")), names=True, delimiter=",")

    fig8 = read_curve("rolo311g_ks2005_fig8_vectors_v1.csv")
    np.testing.assert_array_equal(fig8["wavelength_nm"], table.wave_nm)
    model = moon.rolo311g_reflectance(table.wave_nm, table=table, phase_deg=7,
                                      sun_selenographic_longitude_deg=7)
    rel = fig8["reflectance_unadjusted"] / model - 1
    if len(rel) != 32 or np.max(np.abs(rel)) >= 0.025:
        raise AssertionError("ROLO reflectance departs from the historical Fig. 8 tolerance")

    fig15 = read_curve("liu2021_cubesat_deployable_baffle_pst_fig15_v1.csv")
    angles = fig15["source_angle_deg"]
    np.testing.assert_array_equal(angles, np.arange(28, 90, 2))
    # The plotted 28-degree point constrains the right-hand fit limit.
    # The model refuses an actual pointing at or inside that angle.
    fitted = np.array([pst(max(theta, np.nextafter(28., np.inf))) for theta in angles])
    rms = np.sqrt(np.mean((fig15["pst"] - fitted)**2))
    reference = json.loads(PST_RECORD.read_text(encoding="utf-8"))["residuals"]
    np.testing.assert_allclose(rms, reference["rms"], rtol=1e-10, atol=0)
    try:
        pst(28.)
    except moon.MoonHaloError:
        pass
    else:
        raise AssertionError("A boresight at 28 degrees must be outside the model's usable domain")
    np.testing.assert_array_equal([pst(90.), pst(120.)], [0., 0.])
    return {"rolo_max_absolute_fractional_difference": float(np.max(np.abs(rel))),
            "pst_rms_over_mean_digitized_pst": float(rms / np.mean(fig15["pst"]))}


def cmv_stress(table, pst):
    case = json.loads((DATA / "lunar_stress.json").read_text(encoding="utf-8"))
    observer, lunar, solar = (np.array(case[key]) for key in
                              ("observer_gcrs_km", "moon_gcrs_km", "sun_gcrs_km"))
    lunar_dir, lunar_range = sky.geometry.observer_relative_direction(lunar, observer)
    solar_dir, solar_range = sky.geometry.observer_relative_direction(solar, observer)
    phase, sun_moon_au = moon.phase_angle_deg(lunar_dir, lunar_range, solar_dir, solar_range)
    separation = float(moon.angle_between_deg(case["ray_gcrs"], lunar_dir)[0])
    np.testing.assert_allclose([separation, phase], [28.5, case["phase_deg"]], atol=1e-8, rtol=0)
    if not moon.ROLO_PHASE_DOMAIN_DEG[0] < phase < moon.ROLO_PHASE_DOMAIN_DEG[1]:
        raise AssertionError("This sample requires the fitted ROLO phase domain")
    lit = moon.direct_light_area_weight(observer, lunar_dir, lunar_range, solar_dir, solar_range)
    if lit["method"] != "WHOLE_DISK_LIT" or lit["area_weight"] != 1:
        raise AssertionError("The historical lunar disk should be fully visible and outside the umbra")

    # Penumbral contact is a separate whole-disk test in geocentric geometry.
    d_moon, d_sun = np.linalg.norm(lunar), np.linalg.norm(solar)
    anti_sun_deg = float(moon.angle_between_deg(lunar, -solar)[0])
    contact_rad = (np.arcsin(moon.EARTH_RADIUS_KM / d_moon)
                   + np.arcsin((moon.SUN_RADIUS_KM + moon.EARTH_RADIUS_KM) / d_sun)
                   + np.arcsin(moon.MOON_RADIUS_KM / d_moon))
    clearance = anti_sun_deg - np.degrees(contact_rad)
    np.testing.assert_allclose(clearance, case["penumbra_clearance_deg"], atol=1e-7, rtol=0)
    if clearance <= 0:
        raise AssertionError("Penumbral contact invalidates the fully sunlit reference")

    # Exercise the same public spectrum path as the v27 synthetic Moon, using
    # the retained anomaly-dependent body angles and the original CMV pixel.
    target = dict(case["libration_deg"], moon_centre_to_boresight_deg=separation,
                  moon_observer_unit_gcrs=lunar_dir, observer_moon_distance_km=lunar_range,
                  area_weight=lit["area_weight"])
    geo = dict(sun_direction_gcrs=solar_dir, sun_distance_km=solar_range)
    sun_sed, transmission, qe = sky.instrument("E12")
    radiance = sky.lunar_radiance(target, geo, sun_sed, transmission)
    diameter = case["aperture_mm"] * 1e-3
    pixel_angle = np.radians(case["plate_scale_arcsec"] / 3600.)
    area = np.pi * diameter**2 / 4
    omega = np.tan(pixel_angle)**2
    photon = sky.WAVE * 1e-9 / (h * c)
    measured = area * omega * np.trapezoid(radiance * transmission * qe * photon, sky.WAVE)

    # Independent detector-plane fold of irradiance-ratio PST. Its effective
    # pixel area follows from the historical plate scale and focal length.
    reflectance = moon.rolo311g_reflectance(sky.WAVE, table=table, phase_deg=phase, **case["libration_deg"])
    irradiance = moon.lunar_irradiance_from_reflectance(
        sky.WAVE, reflectance=reflectance, moon_distance_km=lunar_range, moon_sun_distance_au=sun_moon_au,
        solar_sed_w_m2_nm=sun_sed)
    pixel_area = (case["f_number"] * diameter * np.tan(pixel_angle))**2
    detected = np.where(transmission > 0, irradiance * pst(separation) * qe * photon, 0)
    direct = pixel_area * np.trapezoid(detected, sky.WAVE)
    np.testing.assert_allclose(measured, direct, rtol=1e-10, atol=0, err_msg="Independent detector-plane lunar fold")
    diffuse = sum(case["coincident_diffuse_e_per_px_s"].values())
    total = measured + diffuse
    np.testing.assert_allclose([measured, total], [case["moon_e_per_px_s"], case["total_e_per_px_s"]],
                               atol=5e-5, rtol=0)
    return {"separation_deg": separation, "phase_deg": phase, "penumbra_clearance_deg": float(clearance),
            "pst": pst(separation), "lunar_e_per_px_s": float(measured), "diffuse_e_per_px_s": diffuse,
            "total_e_per_px_s": float(total), "lunar_over_diffuse": float(measured / diffuse)}


def paper_scenes():
    """Test the change at fixed July sightline; also check the dated August lunar term."""
    scenes = sky.sightlines()
    synthetic = scenes["strong_lunar"]
    separation = float(moon.angle_between_deg(synthetic["moon"]["moon_observer_unit_gcrs"], synthetic["ray_gcrs"])[0])
    np.testing.assert_allclose(separation, 28.5, atol=1e-8, rtol=0)
    refs = json.loads((sky.DATA / "sky_reference.json").read_text(encoding="utf-8"))
    results = {}
    for band in ("E5", "E12"):
        off = sky.background(scenes["strong"], band)
        on = sky.background(synthetic, band)
        np.testing.assert_array_equal([sky.background(scenes["weak"], band)["lunar_stray"], off["lunar_stray"]], [0, 0])
        for component in off.keys() - {"lunar_stray"}:
            np.testing.assert_allclose(on[component], off[component], rtol=1e-12)
        increase = sum(on.values()) - sum(off.values())
        np.testing.assert_allclose([increase, on["lunar_stray"]], refs["strong_lunar"][band]["lunar_stray"],
                                   rtol=1e-7, atol=0)
        results[band] = {"diffuse_e_per_px_s": sum(off.values()), "lunar_e_per_px_s": on["lunar_stray"],
                         "total_e_per_px_s": sum(on.values())}
    return results


def run():
    table = moon.Rolo311gTable.load_csv(sky.SKY / "rolo311g_kieffer_stone2005_table4_v1.csv",
                                       sky.SKY / "rolo311g_kieffer_stone2005_eq11_v1.csv")
    fit = json.loads(PST_RECORD.read_text(encoding="utf-8"))["fit"]
    pst = moon.Liu2021ExponentialPST(fit["a"], fit["k_per_deg"])
    return {"published_curves": figure_checks(table, pst), "historical_cmv_e12": cmv_stress(table, pst),
            "v27_july_virtual_scmos": paper_scenes()}
