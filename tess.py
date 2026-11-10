"""TESS calibrated image photometry, including the optional six-FFI field comparison."""

import csv
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import map_coordinates
from tess_operators import (
    unit,
    omega_grid,
    smooth_valid,
    sun_directions,
    cutout_background,
    running_median,
    split_orbits,
    t_mag,
    fit_background,
)
from skycore import sky_background as sky

HERE = Path(__file__).resolve().parent / "data/tess"
TABLE = sky.ZodiacalTable.load_csv(HERE / "zodiacal_mu.csv")
COLOUR = np.loadtxt(HERE / "tess_zodiacal_colour.csv", delimiter=",", skiprows=1)


def rows(name):
    with (HERE / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def zodiacal(rays, sun, colour):
    shape = rays.shape[:-1]
    flat = rays.reshape(-1, 3)
    d, b, pole = sky.helioecliptic(flat, sun)
    mu, status = TABLE.lookup(np.degrees(d), np.degrees(abs(b)), at_pole=pole)
    elongation = np.degrees(np.arccos(np.clip(flat @ sun, -1, 1)))
    density = np.interp(elongation, colour[:, 0], colour[:, 1]) * 10 ** (-0.4 * mu)
    return np.where(status == sky.STATUS_ZL_OK, density, np.nan).reshape(shape)


def cutout(path):
    cfg = json.loads((HERE / "example.json").read_text())
    with fits.open(path) as h:
        if (h[0].header["SECTOR"], h[0].header["CAMERA"], h[0].header["CCD"]) != (3, 3, 3):
            raise ValueError("Expected the Sector 3 / camera 3 / CCD 3 example")
        flux = np.asarray(h[1].data["FLUX"], dtype=float)
        times = h[1].data["TIME"].copy()
        quality = h[1].data["QUALITY"].astype(int)
        wcs = WCS(h[2].header)
    med, _, finite = cutout_background(flux)
    valid = np.isfinite(times) & (med > 0) & (finite >= 0.9 * 625) & ((quality & ~2048) == 0)
    ids = np.flatnonzero(valid)
    ids = ids[np.argsort(times[ids])]
    z = np.array(
        [
            zodiacal(np.asarray(cfg["ray"]), sun, COLOUR) * cfg["omega_sr"]
            for sun in sun_directions(times[ids])
        ]
    )
    cat = np.genfromtxt(HERE / "example_gaia.csv", delimiter=",", names=True)
    g, bp, rp = (cat["phot_" + band + "_mean_mag"] for band in ("g", "bp", "rp"))
    tm = t_mag(g, bp, rp)
    zp = cfg["zero_point"]
    star_e = zp["rate_e_per_s"] * 10 ** (-0.4 * (tm - zp["t_mag"]))
    for segment in split_orbits(times[ids]):
        segment = segment[np.isfinite(z[segment])]
        if len(segment) < 20 or np.ptp(times[ids[segment]]) < 2:
            continue
        idx = ids[segment]
        rm = running_median(times[idx], med[idx] / z[segment], 0.5)
        chosen = idx[rm <= rm.min() * 1.03]
        image = np.nanmean(flux[chosen], axis=0)
        image[:, cfg["masked_columns"]] = np.nan
        mid = times[chosen].mean()
        dt = (mid + 2457000 - cfg["J2016_JD"]) / 365.25
        ra = cat["ra"] + np.nan_to_num(cat["pmra"]) * dt / (3.6e6 * np.cos(np.radians(cat["dec"])))
        dec = cat["dec"] + np.nan_to_num(cat["pmdec"]) * dt / 3.6e6
        x, y = wcs.all_world2pix(ra, dec, 0)
        use = (x > -6) & (x < 30) & (y > -6) & (y < 30)
        best, _, _ = fit_background(image, x[use], y[use], star_e[use], tm[use])
        yield len(chosen), mid, best["B"]


def site2(directory):
    raw_colour = np.loadtxt(HERE / "site2_raw_vega_colour.csv", delimiter=",", skiprows=1)
    sources = rows("site2_ffi.csv")
    with np.load(HERE / "site2_model_inputs.npz") as aux:
        wcs = WCS(fits.Header.fromstring(str(aux["wcs_header"]), sep=""))
        x0, y0 = aux["origin_xy"]
        ny, nx = aux["stats_mask"].shape
        yy, xx = np.mgrid[y0 : y0 + ny, x0 : x0 + nx]
        ra, dec = wcs.all_pix2world(xx, yy, 0)
        rays = unit(ra, dec)
        ref_sr = (21 / 206264.80624709636) ** 2
        results = []
        for i, visit in enumerate(dict.fromkeys(r["visit"] for r in sources)):
            corrected = []
            for r in [r for r in sources if r["visit"] == visit]:
                with fits.open(directory / r["file"]) as hdus:
                    hd = hdus[1]
                    if hd.header["DQUALITY"] != 0:
                        raise ValueError(r["file"])
                    src_wcs = WCS(hd.header)
                    sx, sy = src_wcs.all_world2pix(ra, dec, 0, quiet=True)
                    valid = (sx >= 44) & (sx <= 2091) & (sy >= 0) & (sy <= 2047)
                    image = map_coordinates(
                        hd.data.astype(float), [sy, sx], order=1, mode="constant", cval=np.nan
                    )
                    b = np.where(valid & (image > 0), image / omega_grid(src_wcs, sx, sy), np.nan)
                    mid = (hd.header["TSTART"] + hd.header["TSTOP"]) / 2
                    corrected.append(
                        (b - 0.9183 * zodiacal(rays, sun_directions([mid])[0], raw_colour)) * ref_sr
                    )
            target = np.nanmedian(corrected, axis=0) - aux["star_model"][i]
            image, _ = smooth_valid(target, aux["build_mask"][i], 8.0)
            results.append(image)
        mask = aux["stats_mask"]
        obs, model = np.mean(results, axis=0)[mask], aux["galaxy"][mask]
        return np.array(
            [obs.mean(), model.mean(), obs.std(), (obs - model).std(), np.sqrt(np.mean((obs - model) ** 2))]
        )


def run():
    """Check the example from pixels and the independent population reduction from windows."""
    observed = list(cutout(HERE / "ecl_000.0_-72.0_s0003_25x25.fits"))
    windows = rows("common_windows.csv")
    reference = [
        w for w in windows if w["unit"] == "ecl_000.0_-72.0|anti_solar_survey" and w["visit"] == "S3"
    ]
    if len(observed) != len(reference) or not reference:
        raise ValueError("Sector 3 selection returned a different number of windows")
    for (n, mid, b), ref in zip(observed, reference):
        np.testing.assert_equal(n, int(ref["n"]))
        np.testing.assert_allclose([mid, b], [float(ref["mean_btjd"]), float(ref["B"])], rtol=0, atol=5e-6)
        print(f"Sector 3: {n} selected cadences, B = {b:.6f} e-/native pixel/s")
    visits = defaultdict(list)
    for row in windows:
        model = 1.021408688828167 * sum(float(row[key]) for key in ("Z0", "I_iso", "D", "U"))
        visits[row["unit"], row["visit"]].append(model / float(row["B"]) - 1.0)
    units = defaultdict(list)
    for (unit_id, _), errors in visits.items():
        units[unit_id].append(np.median(errors))
    paired = {
        r["unit"]: float(r["model_relative_error_clamp"])
        for r in rows("tess_paired_units.csv")
        if r["population"] == "selected57"
    }
    if set(units) != set(paired):
        raise ValueError("Selected units differ from the paper population")
    unit_residuals = [np.median(units[key]) for key in paired]
    np.testing.assert_allclose(unit_residuals, list(paired.values()), atol=1e-12, rtol=0)
    rms = np.sqrt(np.mean(np.square(unit_residuals)))
    print(
        f"{len(windows)} windows, {len(visits)} visits, {len(units)} units: total sky RMS relative error {100*rms:.5f}%"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ffi-dir", type=Path, help="directory with the six original FFIs listed in data/tess/site2_ffi.csv"
    )
    args = parser.parse_args()
    run()
    if args.ffi_dir:
        values = site2(args.ffi_dir)
        print(
            "Six-FFI field: observed mean, model mean, observed SD, residual SD, RMS (e-/21 arcsec pixel/s):",
            values,
        )
