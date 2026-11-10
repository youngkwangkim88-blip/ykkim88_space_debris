"""Remeasure one historical sky-floor comparison from a local 21 x 21 MAST FITS."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

import tess
from tess_operators import cutout_background, omega_at, running_median, split_orbits, sun_directions, unit

MANIFEST = Path(__file__).resolve().parent / "data/goldens/tess/cutouts.csv"


def remeasure(path, row, *, bundled_crop=False):
    with fits.open(path) as hdus:
        header = hdus[0].header
        if any(int(header[k.upper()]) != int(row[k]) for k in ("sector", "camera", "ccd")):
            raise ValueError("Detector and sector do not match the selected cutout")
        table = hdus[1].data
        flux = np.asarray(table["FLUX"], dtype=float)
        if bundled_crop:
            if path != tess.HERE / "ecl_000.0_-72.0_s0003_25x25.fits" or flux.shape[1:] != (25, 25):
                raise ValueError("The verified crop applies only to the bundled Sector 3 cutout")
            flux = flux[:, 2:-2, 2:-2]
        if flux.ndim != 3 or flux.shape[1:] != (21, 21):
            raise ValueError("The historical measurement uses 21 x 21 native pixels")
        t = np.asarray(table["TIME"], dtype=float)
        flags = np.asarray(table["QUALITY"], dtype=np.int64)
        center = np.asarray(12. if bundled_crop else 10.)
        omega = float(omega_at(WCS(hdus[2].header), center, center))
    med, _, finite = cutout_background(flux)
    keep = np.flatnonzero(np.isfinite(t) & ((flags & ~2048) == 0) & np.isfinite(med)
                          & (med > 0) & (finite >= 0.9 * 21**2))
    keep = keep[np.argsort(t[keep])]
    t, med = t[keep], med[keep]
    ray = unit(float(row["ra_deg"]), float(row["dec_deg"]))
    # A constant absolute zodiacal scale cancels in ratio_floor * Z(epoch).
    z = np.array([tess.zodiacal(ray, sun, tess.COLOUR) * omega for sun in sun_directions(t)])
    choices = []
    for segment in split_orbits(t):
        ix = segment[np.isfinite(z[segment]) & (z[segment] > 0)]
        if len(ix) < 20 or np.ptp(t[ix]) < 2:
            continue
        smooth = running_median(t[ix], med[ix] / z[ix], 0.5)
        j = int(np.argmin(smooth))
        choices.append((smooth[j], t[ix[j]], smooth[j] * z[ix[j]]))
    if not choices:
        raise ValueError("No usable orbit segment for this cutout")
    _, btjd, floor = min(choices, key=lambda item: item[0])
    expected = float(row["measured_floor_e_per_px_s"])
    return dict(position=row["position"], sector=int(row["sector"]), input_file=path.name,
                central_crop_of_bundled_fits=bundled_crop,
                sha256_matches_original=None if bundled_crop else hashlib.sha256(path.read_bytes()).hexdigest() == row["fits_sha256"],
                measured_floor_e_per_px_s=float(floor), historical_floor_e_per_px_s=expected,
                floor_difference_e_per_px_s=float(floor - expected), btjd=float(btjd),
                historical_btjd=float(row["t_floor_btjd"]))


def bundled_example():
    with MANIFEST.open(encoding="utf-8") as stream:
        row = next(r for r in csv.DictReader(stream) if r["position"] == "ecl_000.0_-72.0" and r["sector"] == "3")
    result = remeasure(tess.HERE / "ecl_000.0_-72.0_s0003_25x25.fits", row, bundled_crop=True)
    np.testing.assert_allclose(result["btjd"], result["historical_btjd"], atol=1e-8, rtol=0)
    np.testing.assert_allclose(result["floor_difference_e_per_px_s"], 0, atol=1e-8, rtol=0)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fits", nargs="?", type=Path, help="Extract the FITS under its original name from the MAST ZIP")
    args = parser.parse_args()
    if args.fits is None:
        print(json.dumps(bundled_example(), indent=2))
        return
    with MANIFEST.open(encoding="utf-8") as stream:
        found = [row for row in csv.DictReader(stream) if row["file"] == args.fits.name]
    if len(found) != 1:
        parser.error("Choose a FITS filename from data/goldens/tess/cutouts.csv")
    print(json.dumps(remeasure(args.fits, found[0]), indent=2))


if __name__ == "__main__":
    main()
