# Data

Nothing in this folder is tracked in git except this file and the small `diag_*.json` diagnostics of the surrogates. Everything else is either downloaded from a public source or produced by the scripts. Sizes are those of the paper's run.

## Downloads

| File or folder | Size | Source | Script |
|---|---|---|---|
| `raw/slp.daily.nc` | 0.6 GB | NCEP–NCAR Reanalysis 1 daily-mean SLP, 1948–2023, NOAA PSL | `scripts/download_r1_daily.py` |
| `processed_data/experiments/gaussian_null_era5/era5_msl_daily_r1grid_1979_2022.npy` | 0.3 GB | ERA5 mean sea-level pressure, 6-hourly at 1°, from the public WeatherBench 2 archive; daily means regridded to the R1 grid | `scripts/experiments/gaussian_null/download_era5.py` |
| `processed_data/experiments/beyond_amplitude/{hgt.sfc,land}.nc` | 74 kB | R1 orography and land mask, NOAA PSL (`https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/surface/`) | `curl`, see the main README |
| `processed_data/experiments/blocking/z500_*.npz` | 0.6 GB | R1 daily 500 hPa height, 1948–2023, NOAA PSL | `scripts/experiments/blocking/download_z500.py` |
| six-hourly ERA5 (0.25° and 1.5°) and six-hourly R1 | 5 GB kept, 133 GB streamed | WeatherBench 2 archive; NOAA PSL | `scripts/experiments/era5_6h/fields.py` (written to `$ERA6H_CACHE`) |

NCEP–NCAR Reanalysis 1 data are provided by the NOAA Physical Sciences Laboratory, Boulder, Colorado (https://psl.noaa.gov). ERA5 is produced by ECMWF for the Copernicus Climate Change Service; the copy used here is that of WeatherBench 2 (`gs://weatherbench2/datasets/era5/`), which needs no credentials.

## Produced by the scripts

| Folder | Size | Produced by | Used for |
|---|---|---|---|
| `processed_data/SLP_data_years/` | 6.1 GB | `scripts/regenerate_polar_discs.py` | one 172 × 172 disc per day and sign, 1948–2023 |
| `processed_data/representative_data/` | 4.1 GB | `julia/trajectory_representative_storing.jl` | Ripserer diagrams and representative cycles of those discs |
| `processed_data/prototype/` | 66 MB | `scripts/prototype/extract_features.py` | features of every disc, 1948–2023 |
| `processed_data/experiments/gaussian_null/` | 0.8 GB | `scripts/experiments/gaussian_null/run_surrogates.py`, `run_hybrid.py` | features of R1 and of its 92 surrogates; `diag_*.json` |
| `processed_data/experiments/gaussian_null_era5/` | 0.3 GB + download | the same with `GN_SOURCE=era5` | features of ERA5 and of its 40 surrogates |
| `processed_data/experiments/era5_6h/` | 0.8 GB | `scripts/experiments/era5_6h/` | six-hourly features of ERA5, R1 and their surrogates |
| `processed_data/experiments/beyond_amplitude/` | 1.8 GB | `scripts/experiments/beyond_amplitude/` | per-extremum descriptors and detector tracks |
| `processed_data/experiments/blocking/` | 1.9 GB + download | `scripts/experiments/blocking/` | Z500 fields, features and blocking index |
| `processed_data/paper/` | 7 MB | `scripts/paper/trajectory_analysis.py` | one row per tracked snapshot, for the trajectory figures |

A feature file `feat_<name>_{sup,sub}.npz` holds, for every feature deeper than 2 hPa: `slot` (day or six-hour step), `peak` and `saddle` (Pa), `peak_idx` and `parent_idx` (flat pixel index; −1 if the parent is the rim), `half_km2` (half-depth footprint), and per-field aggregates `agg` (number of extrema, TP1, TP2, largest depth).

## What the figures need

`make_null_figures.py` and `make_figures.py` read `raw/slp.daily.nc`, the feature files and `sample_*.npy` in `processed_data/experiments/gaussian_null/`, and `processed_data/paper/`. `make_paper_figures.py fig06` reads one day of `SLP_data_years/` and `representative_data/`. The tables need none of these.
