# Fragment optics Blender reference

Run `python gold_blender.py` from the package root. NumPy loads the losslessly stored first scene-linear channel of each original orthographic render. The sum of its pixels times the square of the pixel width gives directional optical cross section in m² sr⁻¹. The separate unit-radiance square checks projected area in m².

The static cases cover three orientations each for a plate, block and faceted rod, under a stock Cycles Lambert shader and a prescribed normalized Blinn law. The Blinn emission shader implements the specified radiance and tests its projected integral. Its agreement describes this optical model. Six plate cases also call the package's live `fragment_math.factor`. Cross sections scale with area between the geometrically similar 10 cm public plate and the 14.12 cm reference plate. The fixture uses a 60° phase angle, a scalar reflectance and one achromatic channel.

The separate 0.5 s historical exposure supplies the attitude, viewing geometry, range and weight at each quadrature node. Its nine diffuse midpoints and 27 specular subnodes contribute in linear flux space before conversion to magnitude. This replay tests the photometry conditional on those supplied states. The original render agrees with the cached exposure magnitude to 0.000117 mag. Each reference integral has a relative tolerance of 0.2%. The historical exposure uses the original photometric zero point, recorded explicitly in `gold_blender.py`.

## Optional pixel regeneration

From the package root, with Blender 4.2.23 LTS on PATH, choose two fresh output directories:

```sh
blender -b -P data/goldens/blender/render_blender.py -- rerender-poses --group poses
blender -b -P data/goldens/blender/render_blender.py -- rerender-exposure --group exposure
python gold_blender.py --poses-from rerender-poses --exposure-from rerender-exposure
```

The driver renders 256 × 256 scene-linear images with Cycles on CPU. The camera scale, sampling, shape dimensions, rotations, light and viewing directions, and material parameters are in `cases.json`. It writes EXR images and float32 RGB NumPy arrays. Either command-line override can also be used alone.

## Source

These public inputs and images derive from the A1 editorial report `spectral_blender_20260924`, including `inputs/cases.json`, `inputs/historical_cache_case.json`, `renders/*.npy`, `renders/historical_cache/*.npy` and `matched_driver.py`. The original analytical model lives in `AI_SBSSA_v2/wi/WI-SYNTH-DOMAIN/work/physpop/debris_body_model.py`. The historical cache was produced with revision `249a898f2b0163d737a3b4e4f56320592ef240e5`. Native GGX diagnostics from the source report used a different scattering law and are outside this matched comparison.
