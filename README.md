# Optical detectability of rotating debris

This v30 release accompanies *Effects of exposure time, sky background and spectral response on the space-based optical detection range of rotating debris*. Change a fragment's material, spin, angular speed, exposure, sensor response or background and calculate its expected SNR and SNR-10 range. Measure a sky background directly from a calibrated TESS time series. [한국어 실행 안내](README.ko.md) · [Input sources and reuse](SOURCES.md).

## Run the paper examples

Run `python paper_figures.py` to generate article Figs. 1–12 and supplementary Figs. S9–S12 as PNGs in `figures/`. S9 shares the image used in Fig. 5. The figure index identifies the inputs and starting point of each calculation. [Figure index and commands](paper/README.md) · [v30 release](https://github.com/youngkwangkim88-blip/ykkim88_space_debris/tree/v30).

With Python 3.10, create a virtual environment, activate it, and run the commands below. Installing the pinned packages needs a package index or a wheel cache. The examples then run offline with the files in this directory.

```sh
python -m pip install -r requirements.txt
python paper_figures.py
python reproduce.py
python goldens.py
```

| Command | Inputs and expected result |
| --- | --- |
| `python sky.py` | From each UTC, GCRS observer and sightline, dust-map sample, sky model and optical response, compute the component rates. For E12 the weak, strong and strong plus hypothetical Moon cases give 0.711367, 3.230306 and 3.739400 electrons per native pixel per second. E5 has its own response and unresolved-star colour map. |
| `python fragment.py` | Integrate the visible illuminated faces of the grey 10 × 10 × 1.25 cm plate over the same 100 initial states. At 4 s, E12, the August illumination and 939.5667 arcsec/s, median SNR-10 range is 306.0 km with weak sky, 275.5 km with strong sky and 271.0 km with the added Moon. Also prints 5th and 95th percentiles and median expected SNR at a specified range. |
| `python coadd.py` | Follow one rotating plate through the same four seconds at 329.5233 km. Print SNR for a linear unshifted sum with fixed or optimum weights and for the original time-resolved frames with optimum known-template weights. The fixed-sum SNR is 11.064 for a single 4 s read and 4.743 for eight 0.5 s reads in the weak E12 example. The dark and flat masters are shared within each read scheme. |
| `python tess.py` | Fit the stars and sky in the bundled Sector 3 FITS. Of 1,288 available cadences, the retained window uses 731 and gives 78.947314 electrons per native pixel per second. Separately, reduce the 250 paper window measurements to 98 visits and 57 equal-weight units. The RMS of (model − observed)/observed is 9.80885%. |
| `python factors.py` | Reduce the archived SNR grids at 329.5 km to the first-order shares and speed/phase total effects in Fig. 3. The denominator is total variance of log SNR over the specified grid for that exposure and shape. |
| `python figure11.py --output figure11.csv` | Apply 0.711367 and 3.230306 e⁻/native pixel/s to each of 68 selected 4 s E12 virtual-fragment trails. The median paired range loss is 9.6872% (P5–P95, 8.3176–10.3045%). The gray plate loses 9.9406% of its median range over 100 initial states. [Moment inputs](data/figure36/README.md). |
| `python paper/sources/read_noise_sensitivity.py` | Re-evaluate sky losses at 3.3 and 11.96 e⁻ RMS, including the decrease from 9.94% to 1.55% for the 4 s gray plate. |
| `python reproduce.py` | Runs the preceding examples and a 512-draw Monte Carlo check. For the historical 13 e⁻ read-noise streak the measured-to-analytic weighted variance ratio is 1.06615. The range and coadd examples use the paper's separate 3.3 e⁻ virtual sCMOS. |

### Rebuild a detector frame and check the PSF

For comparison with the earlier v27 r8 work, `figure36.py` uses the extrema of 43 scene-mean sky rates and yields a 7.5844% median range loss. The v30 Fig. 11 uses the annual field-center rates. [Historical rc5 release](https://github.com/youngkwangkim88-blip/ykkim88_space_debris/tree/v0.1.0-rc5).

Run `python figure32.py --output figure32_a.npz` to reconstruct Fig. 5(a) / Fig. S9(a). The input is the archived noiseless target in the first 4 s under the weak E12 sky. The example redraws the fixed pixel maps, 64 darks, 32 flats and the science exposure. It checks all 65 × 305 raw DN and calibrated float32 pixels against the archived frame. The panel uses the nominal 5000 e⁻/pixel flat level. The range calculation in Eq. (5) propagates uncertainty in the measured full-field flat mean and covariance from reusing the same master dark. [Code](figure32.py) · [Frame inputs and provenance](data/figure32/README.md).

With `python figure32.py --from-optics`, the example builds the target from the first sampled rotating plate, optical response and moving PSF, and recomputes the two dated skies. The three panels have expected charges of 1156.69, 1156.69 and 2142.84 e⁻ and SNRs of 11.0638, 9.1170 and 13.7697. Panel (a) also has an archived pixel-by-pixel detector comparison.

For a separate optical check, install `python -m pip install poppy==1.1.2` in the same environment and run `python check_scmos_poppy.py`. On the paper's 2.8 µm virtual sCMOS, the largest absolute pixel difference between the two calculations, divided by the paper image peak in the same window, is 0.5492% (E5) and 0.6808% (E12) for a centred point. For a uniform 0.5 s trail at 130.4423 arcsec/s it is 0.1818% and 0.1978%. These checks compare propagation through the same ideal pupil. The 0.117% and 0.032% values in supplement S3 use the 5.5 µm CMV4000 and Vega weighted by Gaia G. [POPPY comparison](check_scmos_poppy.py).

## Replay the independent comparisons

Run `python goldens.py` for all groups, or append one group name. The default replay reads local observations and Blender pixels and draws new noise from the published seeds. It needs the same environment as the paper examples.

| Group | Independent observable and scope |
| --- | --- |
| `lunar` | Recompute the selected fully sunlit CMV E12 sample (3.42261 e⁻/pixel/s), check its penumbral clearance and compare with an independent detector-plane integral. Verify the separate v27 virtual sCMOS July lunar increment for E5 and E12 and compare ROLO and the baffle fit with published curves. [Inputs, geometry and scope](data/goldens/lunar/README.md). |
| `tess` | For 59 historical 21 × 21 cutouts, start from retained measured sky floors and recalculate zodiacal rates at their epochs. Check geometry and interpolation against independent calculations. The two historical model discrepancies remain visible. [Measurements and product links](data/goldens/tess/README.md). |
| `tess_paper` | Rebuild STScI and ZodiSURF zodiacal rates, diffuse Galactic light and the isotropic term from the selected timestamps, sightlines and TESS band coefficients. The measured sky and unresolved-star rate come from the archived photometry. Compare the resulting 250 windows and all 57 units to the manuscript. [Inputs and aggregation](data/goldens/tess_paper/README.md). |
| `blender` | Integrate the original linear Cycles pixels for 18 pose/law combinations. Six plate comparisons exercise the public photometry function directly. A separate 36-node exposure checks integration in flux before conversion to magnitude. [Optional rerender and model scope](data/goldens/blender/README.md). |
| `motion` | Compare the predicted 2 × 2 angular covariance of the exposure-averaged centroid with 16,000 nonlinear projections of uncertain initial orbital states, each shared across the entire 30–90 s exposure. The spread of photons along the nominal path is reported separately. [Original case and equations](gold_motion.py). |
| `calibration` | Replay 20,000 independent finite-master draws for a 4 s spot on a detector with 13 e⁻ RMS read noise and pixel-to-pixel sensitivity and dark-current variations. For this fixed aperture, empirical/predicted variance is 1.00164 for calibration alone and 0.99052 including science noise. [Sampling and dark reuse](gold_calibration.py). |
| `coadd` | In 30,000 realizations with 3.3 e⁻ RMS read noise and unit sensitivity, move a Gaussian source over three footprints, reuse the same master dark and flat and compare the flux covariance to the variance formula in `coadd.py`. Both the calibration-only and total matrices are tested in squared electrons. [Noise and sampling assumptions](gold_coadd.py). |
| `photometry` | Exercise the original operational SNR measurement on signed residuals from a candidate footprint, including masked pixels and a non-unit flat. Replay finite-reference and four native moving-trail noise controls. [Runnable estimator example and historical inputs](data/goldens/photometry/README.md). |

The paper's SNR-10 distance uses the expected signal in Eq. (5). The reader can apply the forced-photometry estimator in Online Resource 1 to a measured candidate using `operational_photometry.matched_measure` with a calibrated residual, a base variance, a flat and an estimated template.

### Change a physical input

```sh
python fragment.py --band E5 --shape cuboid --material solar_cell --period 10 --exposure 0.5
python fragment.py --period fixed --speed 0 --snr-distance 400
python fragment.py --illumination strong
python coadd.py --shape cuboid --scene strong
```

`--illumination strong` changes the target phase angle and Sun distance to the July date. The default keeps the August target illumination across all three skies to isolate the background effect. Both shapes share a 10 cm side. Surface coefficients and the sampled attitudes are modeling inputs. SNR uses a known trajectory and template. A threshold range is conditional on that model and the chosen angular speed.

To evaluate a new sightline, supply a JSON object with `epoch_utc` (ISO 8601, with UTC offset), `observer_gcrs_km` (geocentric x, y, z), `ray_gcrs` (unit x, y, z) and `I100_MJy_sr` (the SFD 100 μm map sampled at that direction). The weak and strong records in `data/sightlines.json` are templates. Run `python sky.py --scene my_scene.json --band E12`. The command derives the dated Sun and Moon with Astropy's built-in ephemeris. The paper's separate lunar stress example fixes a selected point on the July lunar ellipse at 28.5° and supplies the associated selenographic angles. The code keeps the DGL cap at 25 MJy/sr. Choosing the extrema anew over the whole year requires the mission pointing history.

### TESS data and reproduction scope

The bundled FITS supports the Sector 3 pixel reduction from start to finish. For the optional second field, download the six calibrated FFIs named in `data/tess/site2_ffi.csv` to one directory under their original filenames, then run `python tess.py --ffi-dir PATH`. Its mean measured and modeled rates are 25.622 and 5.814 e⁻/(21 arcsec pixel)/s. This field and statistic are separate from the 57-unit RMS. The 57-unit reduction starts from component photometry in `common_windows.csv`; `common_visits.csv` also identifies the original observational products. Re-extracting every one of those windows from mission images requires the per-visit reduction and masks beyond the Sector 3 worked example.

The component sky rates and fragment quantiles are recomputed from inputs. The reference tables only check the answers. The Fig. 3 calculation starts from saved SNR cells. The short-read coadd experiment preserves the physical total charge and changes the read noise and available timing information. Full-frame image production, annual pointing selection and blind object detection are outside these examples.
