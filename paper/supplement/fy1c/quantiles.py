"""Summarise the simulated FY-1C magnitude deficits supplied beside this script."""
from pathlib import Path

import numpy as np


def main():
    path = Path(__file__).with_name("fy1c_exposures.csv")
    rows = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    quantiles = np.quantile(rows["delta_mag"], [0.10, 0.50, 0.90])
    print(f"{len(rows)} exposures from {len(np.unique(rows['object_index']))} virtual fragments")
    for percentile, value in zip((10, 50, 90), quantiles):
        print(f"P{percentile} {value:+.6f} mag")


if __name__ == "__main__":
    main()
