# Figures for the v30 article and Online Resource 1

From the repository root, install the [pinned requirements](../requirements.txt). Run `python paper_figures.py` for all figures or name labels, for example `python paper_figures.py 5 11 12 S10 S12 --output figures`. Each label writes the PNG used in the article or supplement. S9 uses the same file as Fig. 5. The default redraw works offline. [Drawing entry point](../paper_figures.py) · [Model and observation examples](../README.md) · [Sources and attribution](../SOURCES.md).

| Figure | Output PNG | Scientific input and calculation |
| --- | --- | --- |
| 1 | `geometry.png` | Geometric construction of the sightline, facet angles, cube and plate. |
| 2 | `material_refl.png` | Response-weighted lab measurements in `sources/ijass_figure_inputs.npz` and [diffuse/specular fit parameters](../data/optics/material_laws.csv). |
| 3 | `factor_shares_r8.png` | First-order and total effects on the variance of log SNR. Run `python factors.py` to recompute the plotted shares from the stored [SNR grids](../data/factors). |
| 4 | `paper_ranges_r8.png` | E5/E12 4 s gray-plate range quantiles from the [unrounded ranges](../data/ranges_reference.csv). Run `python fragment.py` for a fresh 100-state calculation. |
| 5, S9 | `first_exposure_E12_r8.png` | Three first-exposure calibrated crops, their expected native-pixel charge profiles and `supplement/paper_frames.csv`. `python figure32.py --from-optics` independently reconstructs the three signals and checks the first detector frame. |
| 6 | `sky_spectral_trade.png` | Paired sky-loss and spectral-response gains from the same unrounded range table for the ten shape–surface combinations. |
| 7 | `speed_anchors.png` | Quantiles over [9,396 image-plane speeds](supplement/catalogue_speed_review/leo_speeds_arcsec_s.csv). Each in-field virtual-fragment exposure has equal weight. |
| 8 | `speed.png` | Fixed and rotating gray-plate median ranges at 80° phase under the stated sky from [the controlled grid](supplement/controlled_10cm.csv). |
| 9 | `reference_coadd.png` | Native E5 reference exposure, half-second target templates, seeded short-read calibration and [readout comparisons](../data/pseudo4_comparison.csv). [Readout and variance propagation](sources/reference_plate_readout.py). |
| 10 | `catalogue_edge_profiles_r8.png` | Two calibrated 81 × 81 crops and their modeled paths. The [plot script](sources/catalogue_edge_profiles_r8.py) checks the displayed pixels against raw DN and calibration masters. |
| 11 | `loss.png` | `python figure11.py --output figure11.csv` solves Eq. (5) for 68 trails from their [charge moments](../data/figure36), using the same two field-center skies as Fig. 4. The median loss is 9.6872%. The historical `figure36.py` uses the extrema of 43 scene-mean skies (7.5844%). |
| 12 | `tess_checks.png` | Masked TESS residual observations and matched Galactic model in the second field. The lower panel uses [57 paired groups](../data/tess/tess_paired_units.csv). To repeat the six-FFI pixel reduction, obtain the [named calibrated images](../data/tess/site2_ffi.csv) and run `python tess.py --ffi-dir PATH`. |
| S10 | `single_read_r8.png` | Faint and bright 0.5 s E12 exposures. The script checks raw DN and masters. For fresh simulated frames run `python paper/sources/single_read_r8.py --release . --archive figures/S10_frames.npz --output figures/S10_fresh.png`. |
| S11 | `production_coadd_example.png` | A 486 km virtual-fragment transit under three readout schemes, from aperture, local annulus, calibrated sums and target expectations in `supplement/production_coadd_cutouts.npz`. |
| S12 | `annual_sky_fields.png` | Four diffuse E12 rates sampled on 129 × 129 local detector-plane positions in each of the two pointings. The two panels share a logarithmic scale and include the actual detector roll. [Sampling and geometry](sources/sample_annual_sky_fields.py) · [Sampled components](supplement/annual_sky_fields.npz). |

## Supplementary calculations

- `python paper/sources/read_noise_sensitivity.py` recalculates both skies, two responses and all ten shape–surface pairs at 0.5 and 4 s. At 4 s and 11.96 e⁻ RMS, the gray-plate E12 medians are 183.27 and 180.42 km.
- `python paper/sources/check_native_variance.py` redraws detector noise 16,384 times for the native 4 s streak. `python paper/supplement/validation/S7/streak_noise_limits.py --output streak_noise.csv` compares analytical variances with Monte Carlo draws for moving sources. `python paper/supplement/fy1c/quantiles.py` reduces the supplied virtual-fragment magnitude differences.
- The redrawing scripts use simulation outputs, code and reduced observations identified in the table. Full campaign frames, orbital elements, full-sky atlases and the source simulation checkout are external. For S12, reevaluation at new sky locations requires the full providers and their external assets. The source revision and two pointing declarations are recorded in the sampling NPZ. The supplied two fields redraw directly without those assets.
