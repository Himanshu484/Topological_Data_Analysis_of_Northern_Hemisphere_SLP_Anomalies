# Six-hourly ERA5 at native resolution, and six-hourly NCEP–NCAR R1

Repeats the null-field comparison of the paper on six-hourly instantaneous
sea-level pressure, 1979–2022 (64,240 fields, no-leap calendar), for ERA5 and,
as the second reanalysis the paper's decision rule needs, for R1 (Section 4.5 of the paper,
Tables 10–13, Figure 16).

| Data set | Grid | Fields | What is computed |
|---|---|---|---|
| `real_0p25` | 0.25° (native), 721 × 721 disc, 28 km pixels | reanalysis | features, coherence, trajectories |
| `real_1p5` | 1.5°, 241 × 241 disc, 83 km pixels | reanalysis | the same |
| `ft_r0…4_1p5` | 1.5° | Gaussian null, 5 realisations | the same |
| `iaaft_r0…4_1p5` | 1.5° | amplitude-adjusted null, 5 realisations | the same |
| `ftw_r0_1p5` | 1.5° | control: Gaussian surrogate of the whole record, no windows or yearly seams | the same |
| `real_r1`, `ft_r0…4_r1`, `iaaft_r0…4_r1`, `ftw_r0_r1` | 2.5° (R1's own grid), 173 × 173 disc, 116 km pixels | NCEP–NCAR R1 4-times-daily SLP from NOAA PSL, and its nulls | the same |

**The nulls are not at 0.25°.** A surrogate needs the whole space–time record of
a seasonal window in memory: 21,296 × 519,840 values at 0.25°, about 130 GB with
its Fourier transform. So ratios to the nulls are formed at 1.5°, and the 0.25°
reanalysis is used to show how much the reanalysis features themselves change
between the two grids.

## Running it

The archive is the public WeatherBench 2 bucket (anonymous access). Reading it
needs `zarr<3` and `gcsfs`; install them in a virtual environment layered on the main one:

```bash
python -m venv --system-site-packages ~/venv_era5
~/venv_era5/bin/pip install 'zarr<3' gcsfs
PY=~/venv_era5/bin/python
export ERA6H_CACHE=/path/with/15GB/free            # scratch space; everything in it can be re-created

cd scripts/experiments/era5_6h
$PY fields.py download-1p5            # 1.5° fields → anomaly memmap (3.8 GB)            ~3 min
$PY fields.py features-1p5            # features of the 1.5° reanalysis                  ~5 min
$PY fields.py clim-0p25               # pass 1 over the 0.25° archive: climatology (3 GB) ~50 min
$PY fields.py features-0p25 7         # pass 2: anomalies and features, streamed          ~60 min
$PY surrogates6h.py --reals 0 1 2 3 4 # nulls and their features                          ~20 min per pair
$PY surrogates6h.py --reals 0 --kinds ftw   # no-seam control                             ~12 min
$PY fields.py download-r1             # R1: 44 yearly files from NOAA PSL (0.85 GB) → anomaly  ~3 min
$PY fields.py features-r1             # features of six-hourly R1                         ~1 min
$PY surrogates6h.py --res r1 --reals 0 1 2 3 4        # R1 nulls                          ~4 min per pair
$PY surrogates6h.py --res r1 --reals 0 --kinds ftw    # R1 no-seam control                ~2 min
$PY analyse6h.py                      # output/experiments/era5_6h/metrics_6h.json        ~15 min
$PY run_checks.py                     # projection and surrogate code against the daily code
cd ../../.. && $PY scripts/paper/make_6h_outputs.py   # Tables 10-13, Figure 16, numbers_6h.json
```

The 0.25° fields are never stored (133 GB); pass 2 writes one feature file per
year to `data/processed_data/experiments/era5_6h/parts/` and can be restarted.
`analyse6h.py` skips data sets already in `metrics_6h.json`; delete the file to
recompute everything. Only the feature files (about 0.5 GB) and the metrics are
kept in the repository folder.

## What differs from the daily analysis

* **Fields** are instantaneous at 00, 06, 12, 18 UTC, not daily means.
* **Climatology**: mean over the 44 years for each calendar day and synoptic hour,
  smoothed with a 31-day running mean over days separately for each hour. The daily
  pipeline used the unsmoothed calendar-day mean; here that would leave about 1 hPa
  of sampling noise in the storm tracks and remove it from every field of that slot.
* **Disc image**: (2c + 1)² pixels with the pole at the centre pixel, c = 120 at 1.5°
  and 360 at 0.25° (c = 86 for 2.5° in the daily analysis). One pixel is 0.75° of
  latitude at 1.5° and 0.25° at 0.25°: the 0.25° image has the meridional resolution
  of the source grid and is coarser than it zonally poleward of 33° N (equal at 61° N
  for the 1.5° image, 66° N for the daily one). `run_checks.py` shows the operator is
  identical to the daily one for c = 86.
* **Surrogates**: same construction (four 121-day windows cross-faded over 30 days,
  common phases at all grid points, 10 IAAFT iterations), one slot per 6 hours.
  Seeds `1000 * 1979 + r`.
* **Tracker**: `src/lifted_tracker.py` with thresholds 5/10 hPa and λ = 0.01 hPa/km,
  for linking radii of 150–1000 km per 6-hour step. A trajectory of n fields is
  counted as n/4 days.
* **Coherence**: within 250 and 500 km at lags of 6 h to 7 d; the chance level is
  the lag of one year. One minus the 6-hour value at 500 km is the tracker-free
  "loss rate" of the manuscript.
* **Count-matched control** (`matched` in the metrics): each surrogate's two thresholds
  are moved so that it has as many features per field as the reanalysis has above 10
  and above 5 hPa; coherence and trajectories are recomputed.
* **Terrain split** (`terrain`): features per field over terrain above/below 1500 m,
  and by synoptic hour over high terrain, from the R1 orography
  `data/processed_data/experiments/beyond_amplitude/hgt.sfc.nc`.
* **Terrain control** (`low`): every feature over terrain above 1500 m removed from
  reanalysis and nulls alike; counts, coherence and trajectories recomputed.
* **Decision rule**: `make_6h_outputs.py` applies the paper's rule over ERA5 1.5° and
  R1 2.5° and both nulls (`verdict` and `verdict_coherence` in `numbers_6h.json`).

## Files

* `grid.py` calendar, paths, projection operator, per-field features
* `fields.py` download, climatology, features of the reanalysis
* `surrogates6h.py` nulls (chunked, memory-mapped; about 4 GB peak) and their features
* `analyse6h.py` all statistics → `metrics_6h.json`
* `run_checks.py` consistency with the daily code
* `../../paper/make_6h_outputs.py` tables and figure of the manuscript section

## Results (every number is in `numbers_6h.json`, written next to the tables)

Ratios are reanalysis / amplitude-adjusted null on the same grid; daily 2.5° value of the same reanalysis in brackets.

| | ERA5 1.5°, lows | R1 2.5°, lows | ERA5 1.5°, highs | R1 2.5°, highs |
|---|---|---|---|---|
| N10 per field | 5.84 | 5.31 | 2.35 | 2.41 |
| N10 ratio | 1.46 (1.34) | 1.35 (1.32) | 0.70 (0.76) | 0.72 (0.76) |
| N30 ratio | 1.82 (1.44) | 1.56 (1.42) | 0.65 (0.75) | 0.68 (0.71) |
| Footprint ratio | 0.58 (0.68) | 0.69 (0.71) | 1.58 (1.37) | 1.52 (1.39) |
| Lost within 6 h, reanalysis / nulls (%) | 10.3 / 17–18 | 11.7 / 18–18.5 | 12.7 / 17–18 | 18.1 / 18–19 |
| same, terrain above 1500 m excluded | 10.1 / 17–18 | 11.0 / 18 | 12.3 / 17–18 | 15.2 / 17.5–18 |
| Mean duration ratio, R = 500 km | 1.51 | 1.46 | 1.39 | 1.13 (1.25 without high terrain) |

By the paper's rule over both reanalyses and both nulls:

* **Established:** every daily difference in number, depth, footprint and nesting (and larger
  than daily, most clearly at 1.5°); lows lost 36–41 % less often within 6 hours; lows'
  mean trajectory duration 1.3–1.6× for R ≥ 375 km; highs' coherence in place from
  12 hours to 7 days (1.7–1.8× at one day, 1.9–2.5× at 2–5 days); highs' share of
  trajectories ≥ 5 d at every radius.
* **Established only with terrain above 1500 m excluded:** highs lost 13–32 % less often;
  highs' mean duration 1.1–1.5× at every radius. Six-hourly R1 has a diurnal artefact
  there (0.56 deep highs per field at 00 UTC, 0.28 at 06 UTC; its nulls 0.41 at all hours).
* **Not established:** any difference in the step of lows (they move as the nulls' extrema
  do; median 179–185 km per 6 h in ERA5 for R ≥ 500 km); the shorter step of highs
  (4–14 % in every comparison, below tolerance in R1); nested share of lows; single-field share.
* **Durations do not converge in the linking radius** (lows: mean 0.4 → 3.6 d for R = 150 →
  1000 km). No absolute lifetime is reported.
* Controls: count-matched thresholds and a no-seam Gaussian surrogate leave the lifetime
  result in place (ratios 1.4–1.7 and 1.2–1.5 for lows at R ≥ 375 km).
* 0.25° against 1.5° (ERA5 reanalysis only): 14 % more lows ≥ 10 hPa, footprints 21 %
  smaller. The 18 % extra highs are all over terrain above 1500 m.
* R1's median step is quantised by its 116 km pixels; use the mean step for R1.
