# Optical detectability of rotating debris

Reproduce selected results of the A1 v27 r6 manuscript on a CPU. Change a fragment's material, spin, angular speed, exposure, sensor response or background and calculate its expected SNR and SNR-10 range. Measure a sky background directly from a calibrated TESS time series. [한국어 실행 안내](README.ko.md) · [Input sources and reuse](SOURCES.md).

## Run the paper examples

With Python 3.10, create a virtual environment, activate it, and run the commands below. Installing the three pinned packages needs a package index or a wheel cache. The examples then run offline with the files in this directory.

```sh
python -m pip install -r requirements.txt
python reproduce.py
```

| Command | Inputs and expected result |
| --- | --- |
| `python sky.py` | From each UTC, GCRS observer and sightline, dust-map sample, sky model and optical response, compute the component rates. For E12 the weak, strong and strong plus hypothetical Moon cases give 0.711367, 3.230306 and 3.739400 electrons per native pixel per second. E5 has its own response and unresolved-star colour map. |
| `python fragment.py` | Integrate the visible illuminated faces of the grey 10 × 10 × 1.25 cm plate over the same 100 initial states. At 4 s, E12, the August illumination and 939.5667 arcsec/s, median SNR-10 range is 306.0 km with weak sky, 275.5 km with strong sky and 271.0 km with the added Moon. Also prints 5th and 95th percentiles and median expected SNR at a specified range. |
| `python coadd.py` | Follow one rotating plate through the same four seconds at 329.5233 km. Print SNR for a linear unshifted sum with fixed or optimum weights and for the original time-resolved frames with optimum known-template weights. The fixed-sum SNR is 11.064 for a single 4 s read and 4.743 for eight 0.5 s reads in the weak E12 example. The dark and flat masters are shared within each read scheme. |
| `python tess.py` | Fit the stars and sky in the bundled Sector 3 FITS. Of 1,288 available cadences, the retained window uses 731 and gives 78.947314 electrons per native pixel per second. Separately, reduce the 250 paper window measurements to 98 visits and 57 equal-weight units. The RMS of (model − observed)/observed is 9.80885%. |
| `python factors.py` | Reduce the archived SNR grids at 329.5 km to all 20 displayed first-order shares in Table 5.1. The denominator is total variance of log SNR over the specified grid for that exposure and shape. |
| `python reproduce.py` | Runs the preceding examples and a 512-draw Monte Carlo check. For the historical 13 e⁻ read-noise streak the measured-to-analytic weighted variance ratio is 1.06615. The range and coadd examples use the paper's separate 3.3 e⁻ virtual sCMOS. |

## Replay the independent comparisons

Run `python goldens.py` for all groups, or append one group name. The default replay reads local observations and Blender pixels and draws new noise from the published seeds. It needs the same three packages as the paper examples.

| Group | Independent observable and scope |
| --- | --- |
| `tess` | For 59 historical 21 × 21 cutouts, start from retained measured sky floors and recalculate zodiacal rates at their epochs. Check geometry and interpolation against independent calculations. The two historical model discrepancies remain visible. [Measurements and product links](data/goldens/tess/README.md). |
| `tess_paper` | Rebuild STScI and ZodiSURF zodiacal rates, diffuse Galactic light and the isotropic term from the selected timestamps, sightlines and TESS band coefficients. The measured sky and unresolved-star rate come from the archived photometry. Compare the resulting 250 windows and all 57 units to the manuscript. [Inputs and aggregation](data/goldens/tess_paper/README.md). |
| `blender` | Integrate the original linear Cycles pixels for 18 pose/law combinations. Six plate comparisons exercise the public photometry function directly. A separate 36-node exposure checks integration in flux before conversion to magnitude. [Optional rerender and model scope](data/goldens/blender/README.md). |
| `motion` | Compare the predicted 2 × 2 angular covariance of the exposure-averaged centroid with 16,000 nonlinear projections of uncertain initial orbital states, each shared across the entire 30–90 s exposure. The spread of photons along the nominal path is reported separately. [Original case and equations](gold_motion.py). |
| `calibration` | Replay the paper's 20,000 independent finite-master draws for a 4 s spot on a detector with pixel-to-pixel sensitivity and dark-current variations. For this fixed aperture, empirical/predicted variance is 1.00164 for calibration alone and 0.99052 including science noise. [Sampling and dark reuse](gold_calibration.py). |
| `coadd` | In a new 30,000-realization exercise, move a Gaussian source over three footprints, re-use the same master dark and flat for the three reads, and compare the resulting flux covariance to the variance formula in `coadd.py`. Both the calibration-only and total matrices are tested in squared electrons. [Noise and sampling assumptions](gold_coadd.py). |
| `photometry` | Exercise the original operational SNR measurement on signed residuals from a candidate footprint, including masked pixels and a non-unit flat. Replay finite-reference and four native moving-trail noise controls. [Runnable estimator example and historical inputs](data/goldens/photometry/README.md). |

The paper's SNR-10 distance uses the expected signal in Eq. (1). The reader can apply Eq. (2) to a measured candidate using `operational_photometry.matched_measure` with a calibrated residual, a base variance, a flat and an estimated template.

### Change a physical input

```sh
python fragment.py --band E5 --shape cuboid --material solar_cell --period 10 --exposure 0.5
python fragment.py --period fixed --speed 0 --snr-distance 400
python fragment.py --illumination strong
python coadd.py --shape cuboid --scene strong
```

`--illumination strong` changes the target phase angle and Sun distance to the July date. The default keeps the August target illumination across all three skies to isolate the background effect. Both shapes share a 10 cm side. Surface coefficients and the sampled attitudes are modeling inputs. SNR uses a known trajectory and template. A threshold range is conditional on that model and the chosen angular speed.

To evaluate a new sightline, supply a JSON object with `epoch_utc` (ISO 8601, with UTC offset), `observer_gcrs_km` (geocentric x, y, z), `ray_gcrs` (unit x, y, z) and `I100_MJy_sr` (the SFD 100 μm map sampled at that direction). The weak and strong records in `data/sightlines.json` are templates. Run `python sky.py --scene my_scene.json --band E12`. The command derives the dated Sun and Moon with Astropy's built-in ephemeris. The paper's separate lunar stress example fixes a selected point on the July lunar ellipse at 28.5° and supplies the associated selenographic angles. The code keeps the v27 DGL cap at 25 MJy/sr. Choosing the extrema anew over the whole year requires the mission pointing history.

### TESS data and reproduction scope

The bundled FITS supports the Sector 3 pixel reduction from start to finish. For the optional second field, download the six calibrated FFIs named in `data/tess/site2_ffi.csv` to one directory under their original filenames, then run `python tess.py --ffi-dir PATH`. Its mean measured and modeled rates are 25.622 and 5.814 e⁻/(21 arcsec pixel)/s. This field and statistic are separate from the 57-unit RMS. The 57-unit reduction starts from component photometry in `common_windows.csv`; `common_visits.csv` also identifies the original observational products. Re-extracting every one of those windows from mission images requires the per-visit reduction and masks beyond the Sector 3 worked example.

The component sky rates and fragment quantiles are recomputed from inputs. The reference tables only check the answers. Table 5.1 starts from saved SNR cells. The short-read coadd experiment preserves the physical total charge and changes the read noise and available timing information. Full-frame image production, annual pointing selection and blind object detection are outside these examples.
