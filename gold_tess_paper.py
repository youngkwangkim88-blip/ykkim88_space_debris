"""Recompute the v27 57-unit TESS comparison from the retained observing geometry and quiet cadences."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from astropy.coordinates import get_sun
from astropy.time import Time
from scipy.constants import h, c
import tess
from tess_operators import unit
from skycore import sky_background, sky_maps
from sky import SKY, WAVE, curve, diffuse_galactic
from skycore import zodisurf_model as zs


DATA = Path(__file__).resolve().parent / "data/goldens/tess_paper"
K_VEGA = 1.021408688828167
NODES_UM = np.round(np.arange(0.30, 1.1001, 0.05), 6)
AREA_M2 = 0.0073984642963915905  # L16 TESS effective area before the single Vega transfer


def fold_band(response_file):
    with Path(response_file).open(encoding="utf-8") as stream:
        numeric = (line for line in stream if line.strip() and not line.lstrip().startswith("#"))
        wave, response = np.loadtxt(numeric, delimiter=",").T
    relative = np.interp(WAVE, wave, response / response.max(), left=0, right=0)
    weight = AREA_M2 * relative * WAVE * 1e-9 / (h*c)
    solar = curve("tsis1_hsrs_1nm_321_1099.csv")
    basis = zs.interpolation_matrix(NODES_UM, WAVE)
    residual = zs.load_isotropic_residual(SKY / "zodisurf_plus_isotropic_residual_v1.json")
    isotropic = zs.isotropic_residual_radiance(WAVE, residual)
    return {
        "solar_node_e_per_s": np.trapezoid(basis * solar * weight, WAVE, axis=1).tolist(),
        "isotropic_e_per_s_sr": float(np.trapezoid(isotropic * weight, WAVE)),
        "north_dgl_e_per_s_sr_per_mjy_sr": float(np.trapezoid(diffuse_galactic(90, 1) * weight, WAVE)),
        "south_dgl_e_per_s_sr_per_mjy_sr": float(np.trapezoid(diffuse_galactic(-90, 1) * weight, WAVE)),
    }

def window_components(field, times, band, parameters):
    """Original per-window model at all quiet epochs, with at most 50 ZodiSURF integrations."""
    ray = unit(float(field["ra_deg"]), float(field["dec_deg"]))
    omega = float(field["omega_sr"])
    epoch = Time(times + 2457000.0, format="jd", scale="tdb")
    rotation = sky_background.ecliptic_rotation_j2000()
    sun_au = get_sun(epoch).cartesian.xyz.to_value("AU").T
    earth_ecliptic = -sun_au @ rotation.T
    sun_ecliptic = -earth_ecliptic / np.linalg.norm(earth_ecliptic, axis=1, keepdims=True)
    ray_ecliptic = rotation @ ray
    projected_ray = ray_ecliptic[:2] / np.linalg.norm(ray_ecliptic[:2])
    projected_sun = sun_ecliptic[:, :2]
    projected_sun = projected_sun / np.linalg.norm(projected_sun, axis=1, keepdims=True)
    delta = np.degrees(np.arccos(np.clip(projected_sun @ projected_ray, -1, 1)))
    beta = np.degrees(np.arcsin(ray_ecliptic[2]))
    elong = np.degrees(np.arccos(np.clip(sun_ecliptic @ ray_ecliptic, -1, 1)))
    mag, status = tess.TABLE.lookup(delta, np.full_like(delta, abs(beta)))
    if not np.all(status == sky_background.STATUS_ZL_OK):
        raise ValueError("ETC table does not cover a retained cadence")
    coloured = np.interp(elong, tess.COLOUR[:, 0], tess.COLOUR[:, 1])
    z2 = float(np.mean(coloured * 10 ** (-0.4*mag)) * omega)

    sample = np.unique(np.rint(np.linspace(0, len(times)-1, min(50, len(times)))).astype(int))
    if np.any(elong[sample] <= 80):
        raise ValueError("Retained sample violates the ZodiSURF solar-angle domain")
    earth = earth_ecliptic[sample]
    kernel, _ = zs.line_of_sight_kernel(np.broadcast_to(ray_ecliptic, earth.shape), earth, earth, NODES_UM, parameters)
    z0 = float(np.mean(kernel @ band["solar_node_e_per_s"]) * omega)
    iso = band["isotropic_e_per_s_sr"] * omega
    _, latitude = sky_background.galactic_from_gcrs(ray)
    north = sky_maps.dgl_north_weight(np.degrees(latitude), half_width_deg=25)
    i100 = float(field["i100_mjy_sr"])
    dgl = min(i100, 25.0) * omega * (north * band["north_dgl_e_per_s_sr_per_mjy_sr"]
                                      + (1-north) * band["south_dgl_e_per_s_sr_per_mjy_sr"])
    return z2, z0, iso, dgl


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def run(*, response_file=None):
    """Return a JSON-ready result; an optional TESS v2 response re-integrates the band coefficients."""
    band = fold_band(response_file) if response_file else json.loads((DATA / "band_folds.json").read_text())
    params = zs.load_parameters(SKY / "zodisurf_model_v1.json")
    fields = read_csv(DATA / "geometry.csv")
    windows = tess.rows("common_windows.csv")
    with np.load(DATA / "tess_l16_cadences.npz", allow_pickle=False) as archive:
        bits = np.cumsum(np.cumsum(archive["second_differences"], dtype="<i8"), dtype="<i8")
        times = bits.view("<f8")
        lengths = archive["window_lengths"]
    if len(fields) != len(windows) or len(lengths) != len(windows) or lengths.sum() != len(times):
        raise ValueError("Window geometry, cadence lengths and observation table disagree")
    epochs = np.split(times, np.cumsum(lengths)[:-1])
    visits = defaultdict(list)
    for field, row, epoch in zip(fields, windows, epochs):
        for key in ("unit", "visit", "window"):
            if row[key] != field[key]:
                raise ValueError(f"Window identity differs for {field}")
        label = (row["unit"], row["visit"], row["window"])
        if len(epoch) != int(row["n"]) or abs(epoch.mean() - float(row["mean_btjd"])) > 1e-9:
            raise ValueError(f"Retained times differ for {label}")
        z2, z0, iso, dgl = window_components(field, epoch, band, params)
        reference = [float(row[key]) for key in ("Z_default", "Z0", "I_iso", "D")]
        np.testing.assert_allclose([z2, z0, iso, dgl], reference, rtol=1e-7, atol=0, err_msg=str(label))
        models = K_VEGA * (np.array([z2, z0, z0 + iso]) + dgl + float(row["U"]))
        observed = float(row["B"])
        # Keep both directions before medians: the reciprocal does not commute with an even-sample median.
        ratios = list(observed / models)
        visits[row["unit"], row["visit"]].append(ratios + [models[2]/observed - 1])

    units = defaultdict(list)
    for (name, _), values in visits.items():
        units[name].append(np.median(values, axis=0))
    paper = {row["unit"]: row for row in read_csv(DATA / "tess_common_units.csv")}
    paired = {r["unit"]: float(r["model_relative_error_clamp"]) for r in tess.rows("tess_paired_units.csv")
              if r["population"] == "selected57"}
    if len(units) != 57 or units.keys() != paper.keys() or units.keys() != paired.keys():
        raise ValueError("Selected unit population differs")
    names = sorted(units)
    values = np.stack([np.median(units[n], axis=0) for n in names])
    expected = [[float(paper[n][k]) for k in ("r_Z2same", "r_Z0", "r_Z1")] + [paired[n]] for n in names]
    np.testing.assert_allclose(values[:, :3], np.asarray(expected)[:, :3], rtol=1e-7, atol=0)
    # The unit residual may be near zero, so use the original relative model-rate budget here.
    np.testing.assert_allclose(1+values[:, 3], 1+np.asarray(expected)[:, 3], rtol=1e-7, atol=0)
    medians = np.median(values[:, :3], axis=0)
    np.testing.assert_allclose(medians, [0.975444, 1.201333, 1.091006], rtol=0, atol=5e-7)
    return {
        "units": len(units), "visits": len(visits), "windows": len(windows), "window_cadence_entries": len(times),
        "median_observed_over_model": dict(zip(("ETC", "ZodiSURF_scattering", "ZodiSURF_plus_isotropic"), medians.tolist())),
        "selected57_total_sky_rms_relative_error": float(np.sqrt(np.mean(values[:, 3]**2))),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response-file", type=Path, help="original TESS instrument response v2.0 CSV")
    args = parser.parse_args()
    print(json.dumps(run(response_file=args.response_file), indent=2))
