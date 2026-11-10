"""Reproduce Table 5.1 from the archived positive, single-exposure SNR cells."""

import csv
from pathlib import Path

import numpy as np

SUPPLEMENT = Path(__file__).resolve().parent / "data/factors"
SHAPES = ("plate", "cuboid")
FACTORS = ("surface_spin", "speed", "phase", "sky", "response")


def archive_snr(filename):
    with np.load(SUPPLEMENT / filename, allow_pickle=False) as archive:
        return archive["snr"]


def main():
    short = archive_snr("factor_snr_0p5s_grid.npz")
    long = np.stack((archive_snr("factor_snr_4s_e5.npz"), archive_snr("factor_snr_grid.npz")), axis=-1)
    expected_shape = (2, 3, 4, 100, 3, 2, 2, 2)
    assert short.shape == long.shape == expected_shape
    with (SUPPLEMENT / "factor_exposure_shares.csv").open(encoding="utf-8", newline="") as report:
        printed = {(row["shape"], row["exposure_s"]): row for row in csv.DictReader(report)}

    print("Shape, exposure (s), first-order shares (%) in table order")
    for index, shape in enumerate(SHAPES):
        for exposure, grid in (("0.5", short), ("4", long)):
            snr = grid[index].reshape(1200, 3, 2, 2, 2)
            assert np.all(np.isfinite(snr)) and np.all(snr > 0)
            y = np.log(snr)
            total_variance = y.var(ddof=0)
            result = []
            for axis, factor in enumerate(FACTORS):
                others = tuple(i for i in range(y.ndim) if i != axis)
                level_means = y.mean(axis=others)
                share = 100 * level_means.var(ddof=0) / total_variance
                reported = printed[shape, exposure][factor]
                decimals = len(reported.split(".")[1])
                value = f"{share:.{decimals}f}"
                if value != reported:
                    raise ValueError(f"{shape}, {exposure} s, {factor}: computed {value}, table {reported}")
                result.append(value)
                if (shape, exposure, factor) == ("plate", "0.5", "response"):
                    example = (level_means, total_variance, share)
            print(", ".join([shape, exposure, *result]))

    means, variance, share = example
    print(
        f"0.5 s plate: mean log SNR E5={means[0]:.8f}, E12={means[1]:.8f}; "
        f"total variance={variance:.8f}; response share={share:.6f}%"
    )


if __name__ == "__main__":
    main()
