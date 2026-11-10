# Figure 3.6. Paired SNR-10 range losses in synthetic catalogue-debris trails

`python figure36.py --output figure36.csv` selects the paper's population, solves Eq. (1) and writes the four threshold ranges and three percentage losses per retained trail. The repository's `reproduce.py` also prints all three distributions. Each trail casts one vote in each distribution.

## Retained inputs

The two CSVs are column extracts of the A1 v27 manuscript supplement. They include all of the original scene/condition and stamp rows. The scripts can therefore distinguish the E12 square campaign, the rectangular pilot, and the high-apogee objects. The public scene and object labels group the original identifiers consistently. The `apogee_le_2000_km` field retains the result of the original catalogue apogee cut. The CSVs contain no orbital elements or precomputed SNR-10 distances.

* `scene_background.csv` gives campaign, synthetic scene, spectral response, exposure `T` (s), frame count and `sky_mean` (e⁻/native pixel/s). The producer divided each frame's diffuse background charge by the full detector area in pixels and by its exposure. The scene rate is the arithmetic mean of those frame rates. It excludes resolved stars and dark current.
* `stamp_moments.csv` gives those campaign keys, the zero-based frame index, object group and apogee class, mean shutter-sampled object range (km), that frame's full-field sky and dark rates (e⁻/pixel/s), and full-detector pixel count. For target-only expected charge `s` at the original range, `src_e, m2, m3, m4` are the sums of the first through fourth powers of `s` over native pixels, and `peak` is its maximum. The script uses the per-frame dark rate recorded in the simulated scenes.

## Selection and calculation

1. Select all 43 square E12 scenes with a 4 s exposure. The minimum and maximum of their 30-frame scene means are 0.7352969 and 2.5978697 e⁻/pixel/s.
2. For the stamp population, the supplement producer examined frames 0, 15 and 29 of each square scene. It retained available catalogue-debris stamps with positive total signal and a finite path entirely within the detector. Select the square E12, 4 s rows with catalogue apogee ≤ 2,000 km. This leaves 68 stamps from 50 objects in 20 scenes. No SNR cut is applied.
3. For each stamp, keep its track, attitude-dependent light history and optics fixed. A brightness multiplier `a = (original range / trial range)²` scales only the target. With fixed-template weights the SNR becomes `a m2 / sqrt(c0 + a c1 + a² c2)`. The shared dark and finite flat contribute to those variance coefficients. Solve the quadratic obtained by setting SNR to 10, take its positive root, and convert it back to kilometres. [Coefficients and solver](../../fragment_math.py)
4. Compute the loss per stamp as `100 (1 − R_after/R_before)`. Weak and strong use the two scene extrema on every stamp. Zero removes the sky component. The original-sky contrast uses the stamp's own frame rate (`sky`), which can differ from its 30-frame scene mean. Sort the resulting losses and use linearly interpolated sample percentiles. For the weak-to-strong comparison, P5, median and P95 are 6.4772%, 7.5844% and 8.0853%. Taking the ratio of the two median distances instead yields 7.1519%.

These estimates are conditional distances for retained synthetic trails. Re-extracting the sky and stamp moments themselves requires the original simulated fields, the target expectation arrays and their catalogue and trajectory records.
