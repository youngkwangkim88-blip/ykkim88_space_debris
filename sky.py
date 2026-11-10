"""Rebuild diffuse and lunar electron rates for the selected v27 sightlines."""

from datetime import datetime
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.constants import h, c
from astropy.utils import iers
from skycore import sky_background as geometry
from skycore import sky_maps, zodisurf_model, moon_halo

iers.conf.auto_download = False
DATA = Path(__file__).resolve().parent / "data"
SKY = DATA / "sky"
WAVE = np.arange(320.0, 1101.0)
FOCAL_M = 0.0366692988883727
APERTURE_M = 0.01833464944418635
AREA = np.pi * (APERTURE_M / 2) ** 2
PIXEL_SR = (2.8e-6 / FOCAL_M) ** 2


def curve(name):
    knots, values = np.loadtxt(DATA / "optics" / name, delimiter=",", skiprows=1).T
    return np.interp(WAVE, knots, values, left=0.0, right=0.0)


def instrument(band):
    transmission = curve("tourmaline_2_0_35_c_transmittance_digitized.csv")
    qe = curve(f"cmv4000_{band.lower()}_qe_digitized.csv")
    solar = curve("tsis1_hsrs_1nm_321_1099.csv")
    return solar, transmission, qe


@lru_cache(maxsize=1)
def zodiacal_model():
    parameters = zodisurf_model.load_parameters(SKY / "zodisurf_model_v1.json")
    residual = zodisurf_model.load_isotropic_residual(SKY / "zodisurf_plus_isotropic_residual_v1.json")
    solar = curve("tsis1_hsrs_1nm_321_1099.csv")
    return zodisurf_model.zodisurf_plus_isotropic_residual_provider(
        WAVE, solar, parameters=parameters, residual=residual
    )


def diffuse_galactic(latitude_deg, i100):
    """Chellew spectrum, blended over ±25 degrees, capped at 25 MJy/sr."""
    if not np.isfinite(i100) or i100 < 0:
        raise ValueError("I100 must be finite and nonnegative in MJy/sr")
    path = SKY / "dgl_spectrum_chellew2022_fig4_recovered_v1.csv"
    with path.open(encoding="utf-8") as stream:
        rows = np.genfromtxt((s for s in stream if not s.startswith("#")), names=True, delimiter=",")
    slopes = []
    for column in ("north_alpha_prime", "south_alpha_prime"):
        valid = np.isfinite(rows[column])
        slopes.append(np.interp(WAVE, rows["lambda_mid_nm"][valid], rows[column][valid]))
    north = sky_maps.dgl_north_weight(np.array([latitude_deg]), half_width_deg=25.0)[0]
    slope = north * 2.1 * slopes[0] + (1 - north) * 1.9 * slopes[1]
    return min(i100, 25.0) * slope * sky_maps.NU_I_NU_100UM_NW_M2_SR_PER_MJY_SR * 1e-9 / WAVE


@lru_cache(maxsize=2)
def unresolved_model(band):
    gmap = SKY / "gaia_dr3_hpx5_unresolved_g_ge_19p0_v1.csv"
    colours = SKY / "gaia_dr3_hpx5_unresolved_g_ge_19p0_colour_factors_v1.csv"
    nodes = SKY / f"gaia_dr3_hpx5_unresolved_{band.lower()}_g_ge_19p0_mean_preserving_nodes_v1.csv"

    def digest(path):
        return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()

    base = sky_maps.UnresolvedCellMap.load_csv(gmap, sha256=digest(gmap), column="mu_ge19p0")
    cells = sky_maps.UnresolvedCellMap.coloured(base, colours, sha256=digest(colours), band=band)
    cells.bind_mean_preserving_nodes(nodes, sha256=digest(nodes))
    return sky_maps.unresolved_cell_provider(
        WAVE,
        cell_map=cells,
        solar_sed_w_m2_nm=curve("tsis1_hsrs_1nm_321_1099.csv"),
        m_sun_g=-26.90588975118956,
        method=sky_maps.CELL_SAMPLING_SMOOTH_RBF_MEAN_PRESERVING,
    )


def lunar_radiance(moon, geo, solar, transmission):
    anomaly_geometry = "moon_observer_unit_gcrs" in moon
    angle = (
        moon["moon_centre_to_boresight_deg"]
        if anomaly_geometry
        else float(np.degrees(geo["moon_separation_rad"][0]))
    )
    if angle <= 28:
        raise ValueError("The baffle law requires lunar centre separation above 28 degrees")
    law = json.loads((SKY / "liu2021_cubesat_deployable_baffle_pst_expfit_v1.json").read_text())["fit"]
    pst_law = moon_halo.Liu2021ExponentialPST(law["a"], law["k_per_deg"])
    pst = pst_law(angle)
    if pst == 0:
        return np.zeros_like(WAVE)
    table = moon_halo.Rolo311gTable.load_csv(
        SKY / "rolo311g_kieffer_stone2005_table4_v1.csv", SKY / "rolo311g_kieffer_stone2005_eq11_v1.csv"
    )
    if not anomaly_geometry:
        provider = moon_halo.moon_stray_light_v2_provider(
            WAVE,
            solar_sed_w_m2_nm=solar,
            pst_model=pst_law,
            optics_transmission=transmission,
            f_number=2.0,
            rolo_table=table,
            reflectance_table=moon_halo.ReflectanceTable.load_csv(
                SKY / "rolo_reflectance_g7_lime2024_fig1_v1.csv"
            ),
            rolo_out_of_domain_fallback=moon_halo.SOURCE_ROUTE_A7_GREY,
        )
        result = provider(dict(geo, moon_boresight_separation_deg=angle))
        radiance = result["radiance"][0]
        if not np.isfinite(radiance).all():
            raise ValueError(f"Unsupported dated lunar direction: {result['status']}")
        return radiance
    phase, distance_au = moon_halo.phase_angle_deg(
        moon["moon_observer_unit_gcrs"],
        moon["observer_moon_distance_km"],
        geo["sun_direction_gcrs"],
        geo["sun_distance_km"],
    )
    reflectance = moon_halo.rolo311g_reflectance(
        WAVE,
        table=table,
        phase_deg=phase,
        **{
            key: moon[key]
            for key in (
                "sun_selenographic_longitude_deg",
                "observer_selenographic_latitude_deg",
                "observer_selenographic_longitude_deg",
            )
        },
    )
    irradiance = moon_halo.lunar_irradiance_from_reflectance(
        WAVE,
        reflectance=reflectance,
        moon_distance_km=moon["observer_moon_distance_km"],
        moon_sun_distance_au=distance_au,
        solar_sed_w_m2_nm=solar,
    )
    conv = moon_halo.irradiance_to_sky_equivalent_conversion(transmission, f_number=2.0)
    return irradiance * conv * pst * moon["area_weight"]


def background(scene, band):
    """Return component rates in electrons per native sCMOS pixel per second."""
    solar, transmission, qe = instrument(band)
    geo = geometry.sky_geometry(
        np.array([scene["ray_gcrs"]]),
        epoch=datetime.fromisoformat(scene["epoch_utc"]),
        observer_gcrs_km=np.asarray(scene["observer_gcrs_km"]),
    )
    model = zodiacal_model()
    z = model(geo)
    if z["status"][0] != geometry.STATUS_ZL_OK:
        raise ValueError(f"Unsupported zodiacal geometry: {z['status'][0]}")
    iso = model.isotropic_residual_radiance_w_m2_sr_nm
    unresolved = unresolved_model(band)(geo["gal_l_deg"], geo["gal_b_deg"])
    spectra = {
        "zodiacal_Z0": z["radiance"][0] - iso,
        "isotropic": iso,
        "dgl": diffuse_galactic(geo["gal_b_deg"][0], scene["I100_MJy_sr"]),
        "unresolved_stars": unresolved["radiance"][0],
        "lunar_stray": lunar_radiance(scene.get("moon", {}), geo, solar, transmission),
    }
    electron_weight = transmission * qe * WAVE * 1e-9 / (h * c) * AREA * PIXEL_SR
    return {name: float(np.trapezoid(spectrum * electron_weight, WAVE)) for name, spectrum in spectra.items()}


def sightlines():
    return json.loads((DATA / "sightlines.json").read_text())


def run():
    reference = json.loads((DATA / "sky_reference.json").read_text())
    for name, scene in sightlines().items():
        for band in ("E5", "E12"):
            result = background(scene, band)
            for component, rate in result.items():
                np.testing.assert_allclose(
                    rate,
                    reference[name][band][component],
                    rtol=1e-7,
                    atol=1e-10,
                    err_msg=f"{name}/{band}/{component}",
                )
            print(
                f"{name:14} {band:3} {sum(result.values()):.6f} e-/pixel/s  "
                + "  ".join(f"{k}={v:.5f}" for k, v in result.items())
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene", type=Path, help="JSON object with epoch_utc, observer_gcrs_km, ray_gcrs and I100_MJy_sr"
    )
    parser.add_argument("--band", choices=("E5", "E12"), default="E12")
    args = parser.parse_args()
    if args.scene:
        custom = json.loads(args.scene.read_text(encoding="utf-8"))
        ray = np.asarray(custom["ray_gcrs"], dtype=float)
        if ray.shape != (3,) or not np.isclose(np.linalg.norm(ray), 1, rtol=0, atol=1e-7):
            parser.error("ray_gcrs must be a three-component unit vector")
        # A custom date and ray always use their dated Moon.
        custom.pop("moon", None)
        values = background(custom, args.band)
        print(json.dumps({**values, "total": sum(values.values())}, indent=2))
    else:
        run()
