"""Replay A1 physics and observation goldens from local data."""

import argparse
import json

import gold_blender
import gold_calibration
import gold_coadd
import gold_motion
import gold_photometry
import gold_tess
import gold_tess_paper

CHECKS = {
    "tess": gold_tess.run,
    "tess_paper": gold_tess_paper.run,
    "blender": gold_blender.run,
    "motion": gold_motion.run,
    "calibration": gold_calibration.run,
    "coadd": gold_coadd.run,
    "photometry": gold_photometry.run,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group", nargs="?", choices=["all", *CHECKS], default="all")
    args = parser.parse_args()
    selected = CHECKS if args.group == "all" else {args.group: CHECKS[args.group]}
    results = {name: check() for name, check in selected.items()}
    print(json.dumps(results, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
