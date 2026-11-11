"""Run the bundled, offline A1 examples. Execute from any working directory."""

import json
import numpy as np
import sky
import fragment
import tess
import factors
import coadd
import figure11
from noise import weak_streak


def monte_carlo():
    settings = json.loads((sky.DATA / "noise.json").read_text())
    with np.load(sky.DATA / "weak_streak.npz") as data:
        _, draws, measured, expected = weak_streak(data, settings)
    ratio = measured / expected
    np.testing.assert_allclose(ratio, 1.06615, rtol=0, atol=0.000005)
    print(f"Legacy 13 e- noise example, {draws} draws: measured/analytic variance = {ratio:.5f}")


def main():
    sky.run()
    fragment.run()
    coadd.run()
    figure11.run()
    tess.run()
    factors.main()
    monte_carlo()


if __name__ == "__main__":
    main()
