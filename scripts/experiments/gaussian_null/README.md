# Null fields for daily SLP anomalies

The core of the paper (Sections 2.7 and 4.4, Tables 3–9 and 15, Figures 4 and 12–15, 17, Appendix C): persistence features of the reanalysis and of surrogate fields, and the statistics that compare them.

All scripts run from this folder. `GN_SOURCE=era5` switches every script from NCEP–NCAR R1 (1948–2023, main period 1979–2023) to ERA5 (1979–2022); data and outputs then go to the folders with the suffix `_era5`.

## Method

1. **Anomaly and image** (`common.py`). Northern Hemisphere daily SLP on the 2.5° grid, 29 February dropped, calendar-day mean removed. The field is mapped by bilinear interpolation onto a 172 × 172 azimuthal-equidistant disc; the map is linear and is stored once as a sparse 23,215 × 5,328 operator. `validate_projection.py` checks it against the stored discs.
2. **Persistence and footprints** (`uf_basin.c`, `basins.py`). Union–find sweep with 8-connectivity and the elder rule; the rim of the disc is the eldest region. For each feature it returns the extremum, the saddle, the parent, the area of the merge-tree basin, and the area of the region above any level around the extremum (the half-depth footprint uses the level extremum − depth/2). Pixel areas are exact for the projection. `validate_basins.py` compares with a Python implementation and with brute-force labelling. The C library is compiled on first use.
3. **Surrogates** (`surrogates.py`, `run_surrogates.py`). The anomaly is standardised by a seasonally varying standard deviation and split into four overlapping 121-day seasonal windows, cross-faded over 30 days.
   - `ft` (Gaussian null): the same random phase at every grid point for each frequency of a window, which keeps every auto- and cross-spectrum (Prichard and Theiler).
   - `iaaft` (amplitude-adjusted null): multivariate iterated amplitude adjustment (Schreiber and Schmitz), 10 iterations, ending on the rank remap to the observed values of each grid point.
   - `ftg`: Gaussian surrogate of the whole record without windows, a control for the yearly seams.
   - `hyb` (`run_hybrid.py`): the 1948–78 standardised structure with the mean and variance of 1979–2023, for the 1979 step.
   Each realisation writes `feat_<kind>_<y0>_<y1>_r<r>_{sup,sub}.npz` and `diag_<...>.json` (skewness, variance, autocorrelation, zonal correlation). Seeds are `1000 × y0 + r`.
4. **Statistics** (`analyse.py` → `metrics.json`): features per day by depth and season, total persistence, footprints (also at matched count), nesting, area–depth exponent, next-day coherence, nearest-neighbour track statistics. `rank_test.py` adds rank probabilities.
5. **Robustness**
   - `half_depth_sensitivity.py`: footprint at 1/3, 1/2 and 2/3 of the depth (Table 8).
   - `track_sensitivity.py`: linking radius and thresholds for the nearest-neighbour tracker.
   - `lagged_coherence.py`: tracker-free persistence, with a lag of 365 days as the chance level (Table 9).
   - `terrain_check.py`: everything again without features over terrain above 1500 m (Appendix C).

## Commands

See step 4 of the main README. One surrogate pair takes about 7 minutes and 1.5 GB; `analyse.py` caches each realisation in `output/experiments/gaussian_null/cache/` and uses `GN_WORKERS` processes.

## Outputs

`output/experiments/gaussian_null{,_era5}/`: `metrics.json`, `rank_test.json`, `half_depth_sensitivity.json`, `track_sensitivity.json`, `lagged_coherence.json`, `terrain_check.json`.
