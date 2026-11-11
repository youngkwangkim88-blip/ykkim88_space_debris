"""Apply the two annual field-center sky rates to the 68 retained 4 s E12 trails (Fig. 11)."""

import argparse
import csv
from pathlib import Path

import numpy as np

from figure36 import square_four_second_rows
from fragment_math import threshold

DATA = Path(__file__).resolve().parent / "data"

def calculate():
    stamps = [row for row in square_four_second_rows("stamp_moments.csv")
              if row["apogee_le_2000_km"] == "1"]
    keys = [(r["scene"], int(r["frame"]), r["object_group"]) for r in stamps]
    assert len(keys) == len(set(keys)) == 68
    assert len({k[0] for k in keys}) == 20 and len({k[2] for k in keys}) == 50
    with (DATA / "ranges_reference.csv").open(encoding="utf-8") as stream:
        plate = {r["case"]: r for r in csv.DictReader(stream) if r["band"] == "E12"
                 and r["shape"] == "plate" and r["material"] == "grey" and float(r["T"]) == 4}

    def column(name):
        return np.array([float(row[name]) for row in stamps])

    moments = [column(name) for name in ("src_e", "m2", "m3", "m4")]
    ranges = {}
    for label, case in (("faint", "weak_date"), ("bright", "strong_fixed")):
        rate = float(plate[case]["b"])
        args = (column("range_km"), column("T"), rate, column("n_pix"), column("dark_rate"))
        distance, scale = threshold(moments, *args)
        assert np.all(scale * column("peak") + (rate + column("dark_rate")) * column("T") < 7400)
        quarter_moments = [m * 0.25**n for n, m in enumerate(moments, start=1)]
        half_distance, _ = threshold(quarter_moments, *args)
        np.testing.assert_allclose(half_distance, distance / 2, rtol=1e-11)
        ranges[label] = distance
    plate_loss = 100 * (1 - float(plate["strong_fixed"]["p50"]) / float(plate["weak_date"]["p50"]))
    return keys, ranges["faint"], ranges["bright"], plate_loss

def run(output=None):
    keys, faint, bright, plate_loss = calculate()
    loss = 100 * (1 - bright / faint)
    q = np.percentile(loss, [5, 50, 95])
    print("Fig. 11, 68 individual streaks: P05 / median / P95 range loss (%) = "
          + " / ".join(f"{v:.4f}" for v in q))
    print(f"Loss of the gray plate's median range across 100 states: {plate_loss:.4f}%")
    if output is not None:
        with Path(output).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("scene", "frame", "object_group", "faint_km", "bright_km", "loss_pct"))
            writer.writerows((*key, f, b, l) for key, f, b, l in zip(keys, faint, bright, loss))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write the paired ranges and percentage losses")
    run(parser.parse_args().output)
