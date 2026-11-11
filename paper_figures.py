"""Draw all article and supplementary figures or select figures by their paper labels."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paper.sources import annual_sky_fields, catalogue_edge_profiles_r8, redraw_figures
from paper.sources import reference_plate_readout, single_read_r8

FIGURES = {
    "1": ("geometry", redraw_figures.geometry),
    "2": ("material_refl", redraw_figures.specimens),
    "3": ("factor_shares_r8", redraw_figures.shares),
    "4": ("paper_ranges_r8", redraw_figures.ranges),
    "5": ("first_exposure_E12_r8", redraw_figures.crop_sequence),
    "6": ("sky_spectral_trade", redraw_figures.sky_response_trade),
    "7": ("speed_anchors", redraw_figures.virtual_speeds),
    "8": ("speed", redraw_figures.speed_ranges),
    "9": ("reference_coadd", reference_plate_readout.draw),
    "10": ("catalogue_edge_profiles_r8", catalogue_edge_profiles_r8.draw),
    "11": ("loss", redraw_figures.paired_sky_ranges),
    "12": ("tess_checks", redraw_figures.tess),
    "S10": ("single_read_r8", single_read_r8.draw),
    "S11": ("production_coadd_example", redraw_figures.exposure_sums),
    "S12": ("annual_sky_fields", annual_sky_fields.draw),
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figure", nargs="*", metavar="FIG", help="1–12 or S9–S12 (S9 reuses Fig. 5)")
    parser.add_argument("--output", type=Path, default=Path("figures"), help="destination directory")
    args = parser.parse_args()
    requested = [item.upper() for item in args.figure] or list(FIGURES)
    labels = list(dict.fromkeys("5" if item == "S9" else item for item in requested))
    for key in labels:
        if key not in FIGURES:
            parser.error(f"Unknown figure {key!r}. Choose 1–12 or S9–S12.")
    args.output.mkdir(parents=True, exist_ok=True)
    redraw_figures.OUT = args.output
    for key in labels:
        filename, draw = FIGURES[key]
        output = args.output / (filename + ".png")
        style = {} if key in {"S10", "S12"} else redraw_figures.STYLE
        with plt.rc_context(style):
            if key in {"2", "12"}:
                with np.load(redraw_figures.EXTRA, allow_pickle=False) as aux:
                    draw(aux)
            elif key == "9":
                draw(redraw_figures.DATA, output)
            elif key == "S12":
                draw(annual_sky_fields.default_data(), output)
            elif key in {"10", "S10"}:
                draw(output)
            else:
                draw()
        print(f"Fig. {key}: {output}")

if __name__ == "__main__":
    main()
