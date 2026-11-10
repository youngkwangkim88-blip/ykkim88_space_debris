"""Recompute Figure 3.6 from scene skies and native-pixel stamp moments."""

import argparse
import csv
from pathlib import Path

import numpy as np

from fragment_math import threshold


DATA = Path(__file__).resolve().parent / "data/figure36"
CONTRASTS = (("weak", "strong"), ("zero", "strong"), ("zero", "own"))


def square_four_second_rows(filename):
    with (DATA / filename).open(encoding="utf-8", newline="") as stream:
        return [row for row in csv.DictReader(stream)
                if row["campaign"] == "square" and row["band"] == "E12" and float(row["T"]) == 4]


def run(output=None):
    scenes = square_four_second_rows("scene_background.csv")
    sky = {row["scene"]: float(row["sky_mean"]) for row in scenes}
    assert len(sky) == len(scenes)
    weak, strong = min(sky.values()), max(sky.values())
    stamps = [row for row in square_four_second_rows("stamp_moments.csv")
              if row["apogee_le_2000_km"] == "1"]
    n_objects = len({r["object_group"] for r in stamps})
    n_scenes = len({r["scene"] for r in stamps})
    print(f"{len(scenes)} scene means: weak {weak:.7f}, strong {strong:.7f} e-/pixel/s")
    print(f"{len(stamps)} eligible stamps from {n_objects} objects in {n_scenes} scenes")

    def values(name):
        return np.array([float(row[name]) for row in stamps])

    moments = [values(name) for name in ("src_e", "m2", "m3", "m4")]
    ranges = {}
    # Own sky is the full-field mean for the stamp's frame. Weak and strong
    # are extrema of the per-scene means including scenes without stamps.
    for case, rate in (("zero", 0.0), ("weak", weak), ("strong", strong), ("own", values("sky"))):
        distance, scale = threshold(moments, values("range_km"), values("T"), rate,
                                    values("n_pix"), dark=values("dark_rate"))
        peak_e = scale * values("peak") + (rate + values("dark_rate")) * values("T")
        assert np.all(peak_e < 7400)
        ranges[case] = distance

    columns = {f"R10_{case}_km": distance for case, distance in ranges.items()}
    for before, after in CONTRASTS:
        loss = 100 * (1 - ranges[after] / ranges[before])
        columns[f"loss_{before}_to_{after}_pct"] = loss
        q = np.percentile(loss, [0, 5, 50, 95, 100], method="linear")
        print(f"{before} -> {after} range loss (%), min / P5 / median / P95 / max: "
              + " / ".join(f"{v:.4f}" for v in q))
    ratio_of_medians = 100 * (1 - np.median(ranges["strong"]) / np.median(ranges["weak"]))
    print(f"Separately, loss from the two median ranges: {ratio_of_medians:.4f}%")
    if output is not None:
        keys = ("scene", "frame", "object_group")
        with Path(output).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, [*keys, *columns])
            writer.writeheader()
            for i, stamp in enumerate(stamps):
                row = {k: stamp[k] for k in keys}
                row.update({k: v[i] for k, v in columns.items()})
                writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write per-stamp ranges (km) and paired percentage losses")
    run(parser.parse_args().output)
