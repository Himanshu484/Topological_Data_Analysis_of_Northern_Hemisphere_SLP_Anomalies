# Topological data analysis of Northern Hemisphere sea-level pressure anomalies

Code for the paper

> H. Yadav, G. D. Charó and D. Faranda, *Topological Data Analysis of Northern Hemisphere SLP Anomalies: Identifying and Tracking Atmospheric Pressure Systems and Testing Them against Null Fields* (arXiv:2511.19938).

The paper identifies lows and highs of daily and six-hourly sea-level-pressure (SLP) anomalies as the degree-one persistent-homology classes of the field (its local extrema, ranked by prominence), tracks them with a lifted Wasserstein matching, and compares their statistics with two ensembles of surrogate fields that keep the seasonal space–time covariance of the anomalies (Gaussian null) and, in addition, the distribution at every grid point (amplitude-adjusted null). It uses two reanalyses, NCEP–NCAR Reanalysis 1 (R1) and ERA5.

Every table and every data figure of the paper is written by a script in this repository. The manuscript itself is not part of the repository.

## Three ways to reproduce

| Level | What you get | What you need | Time |
|---|---|---|---|
| 1. Tables from stored results | All 16 tables, and every number quoted from them | This repository only | 1 minute |
| 2. Figures | All 17 figures | Level 1 plus the feature files and the R1 daily file (`data/README.md`) | 5 minutes |
| 3. Everything from the reanalyses | Feature files, surrogates, statistics | Public downloads (about 3 GB kept, 133 GB streamed), 16 GB of memory, 20 GB of disk | one to two days of computing |

Level 1 works after `git clone`: the summaries in `output/` and the small `diag_*.json` files in `data/` are tracked, and the table scripts read nothing else.

```bash
pip install -r requirements.txt
python scripts/paper/make_null_tables.py      # null-field fragments and the decision rule
python scripts/paper/make_tables.py           # Tables 1-9, 14, 15
python scripts/paper/make_6h_outputs.py       # Tables 10-13 and Figure 16
python scripts/paper/make_terrain_outputs.py  # Table C.16
```

Tables and figures are written to `../manuscript/{tables,figures}` if that folder exists next to the repository, otherwise to `paper_outputs/` inside it; set `MANUSCRIPT_DIR` to choose another place (`scripts/paper/paths.py`).

## Environment

Python 3.11 with the packages of `requirements.txt`. A C compiler is needed once: `scripts/experiments/gaussian_null/basins.py` compiles `uf_basin.c` on first use. Level 3 also needs Julia with `Ripserer`, `NPZ`, `JSON` and `Plots` for one step, and a separate virtual environment with `zarr<3` and `gcsfs` for the ERA5 downloads (see `requirements.txt`).

```bash
python -m pytest -q tests/        # 35 tests, a few seconds; those that need data are skipped without it
```

The paper's results were produced on macOS (Apple silicon, 16 GB). Surrogates are seeded, so reruns give the same realisations.

## Which script makes what

| Paper item | Script | Reads |
|---|---|---|
| Figs 1, 2, 3, 5, 6 | `scripts/make_paper_figures.py` | Fig. 6: stored discs and Ripserer output |
| Figs 4, 12, 13, 14, 17 | `scripts/paper/make_null_figures.py` | daily feature files, R1 daily file |
| Figs 7–11, 15 | `scripts/paper/make_figures.py` | `output/paper/`, `data/processed_data/paper/`, R1 daily file |
| Fig. 16, Tables 10–13 | `scripts/paper/make_6h_outputs.py` | `output/experiments/era5_6h/metrics_6h.json` |
| Tables 1, 2, 4, 5, 14 | `scripts/paper/make_tables.py` | `output/paper/analysis_{R1,ERA5}.json`, detector results |
| Tables 3, 6–9, 15 | `scripts/paper/make_null_tables.py`, then `make_tables.py` | `output/experiments/gaussian_null{,_era5}/` |
| Table C.16 | `scripts/paper/make_terrain_outputs.py` | `terrain_check.json` of both reanalyses |
| Blocking test (Section 5.3, numbers in the text) | `scripts/experiments/blocking/` | `output/experiments/blocking/*.txt` |
| Verification of the duality statement (Section 2.5) | `scripts/prototype/extract_features.py` | stored discs and Ripserer output |

Each table script also writes a `numbers_*.json` file next to the tables with every value used in the text, and applies the paper's decision rule ("established" differences).

## Level 3: the full pipeline

Run from the repository root. `PY=python` in the environment above; `PYE` is the Python of the ERA5 download environment. Times are for 6–8 cores.

**1. Data** (details and sizes in `data/README.md`)

```bash
$PY scripts/download_r1_daily.py                                   # R1 daily SLP 1948-2023, 0.6 GB, 2 min
$PYE scripts/experiments/gaussian_null/download_era5.py            # ERA5 daily on the R1 grid, 0.3 GB, 12 min
mkdir -p data/processed_data/experiments/beyond_amplitude
for f in hgt.sfc.nc land.nc; do curl -o data/processed_data/experiments/beyond_amplitude/$f \
    https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/surface/$f; done
```

**2. Stored discs and reference persistence** (used by Fig. 6, by the verification of Section 2.5, and by the 1948–2023 feature table of steps 3 and 5)

```bash
$PY scripts/regenerate_polar_discs.py                # 55,480 polar discs, 6.1 GB
julia julia/trajectory_representative_storing.jl     # Ripserer diagrams and representatives, 4.1 GB; the slowest single step
```

**3. Features 1948–2023 by union-find, checked against Ripserer** (10 min)

```bash
PY=$PY bash scripts/prototype/run_all.sh
```

**4. Null fields, daily** (`scripts/experiments/gaussian_null/`; run from that folder). Every script takes `GN_SOURCE=era5` to switch from R1 to ERA5.

```bash
$PY run_surrogates.py --real
$PY run_surrogates.py --period 1979 2023 --kinds ft iaaft --reals $(seq 0 39)    # 7 min per pair
$PY run_surrogates.py --period 1948 1978 --kinds ft iaaft --reals 0 1 2 3
$PY run_surrogates.py --period 1979 2023 --kinds ftg --reals 0                   # control without yearly seams
$PY run_hybrid.py --reals 0 1 2                                                  # for the 1979 step
GN_SOURCE=era5 $PY run_surrogates.py --real
GN_SOURCE=era5 $PY run_surrogates.py --period 1979 2022 --kinds ft iaaft --reals $(seq 0 19)
$PY analyse.py;  GN_SOURCE=era5 $PY analyse.py;  $PY rank_test.py
$PY half_depth_sensitivity.py --reals 0 1 2                                      # 25 min
$PY track_sensitivity.py 5;   GN_SOURCE=era5 $PY track_sensitivity.py 5
$PY lagged_coherence.py 5;    GN_SOURCE=era5 $PY lagged_coherence.py 5
$PY terrain_check.py;         GN_SOURCE=era5 $PY terrain_check.py                # 8 min each
$PY validate_basins.py;       $PY validate_projection.py                         # checks
```

**5. Detector comparison and blocking test**

```bash
PY=$PY bash scripts/experiments/beyond_amplitude/run_all.sh        # 40 min
# blocking: see scripts/experiments/blocking/README.md
```

**6. Six-hourly fields** — see `scripts/experiments/era5_6h/README.md` (about 4 hours, most of it two passes over the 0.25° ERA5 archive).

**7. Trajectories, tables, figures**

```bash
$PY scripts/paper/trajectory_analysis.py;  GN_SOURCE=era5 $PY scripts/paper/trajectory_analysis.py    # 5 min each
$PY scripts/paper/make_null_tables.py && $PY scripts/paper/make_tables.py
$PY scripts/paper/make_6h_outputs.py && $PY scripts/paper/make_terrain_outputs.py
$PY scripts/paper/make_null_figures.py && $PY scripts/paper/make_figures.py && $PY scripts/make_paper_figures.py
```

## Layout

```
src/
  lifted_tracker.py        lifted Wasserstein tracker with two depth thresholds (the tracker of the paper)
  tda_metrics.py           Wasserstein distance between diagrams with an explicit ground metric
  globe_visualization.py   the azimuthal-equidistant disc projection
  disc_utils.py            no-leap calendar and pixel <-> (lat, lon) helpers
  paper_figures.py         the pedagogical figures
scripts/
  download_r1_daily.py, regenerate_polar_discs.py, make_paper_figures.py
  prototype/               union-find persistence of one disc; features of all stored discs; nearest-neighbour linking
  experiments/
    gaussian_null/         surrogates, persistence and footprints (C), statistics, checks: the core of the paper
    era5_6h/               the same on six-hourly ERA5 (0.25 and 1.5 degrees) and six-hourly R1
    beyond_amplitude/      topological depth against other detection criteria; how trajectories end
    blocking/              test against a Tibaldi-Molteni blocking index on 500 hPa heights
  paper/                   trajectory statistics, tables, figures
julia/                     Ripserer.jl diagrams and representative cycles of the stored discs
tests/
output/                    result summaries (tracked)
data/                      downloads and feature files (not tracked, see data/README.md)
```

The folder names `prototype`, `gaussian_null` and `beyond_amplitude` are historical. Each folder has a README with its method and commands.

## Conventions worth knowing

- **Calendar:** 365-day no-leap calendar; 29 February is dropped. A "slot" is a day (or a six-hour step) counted from 1 January of the first year.
- **Image:** azimuthal-equidistant disc, North Pole at the centre pixel, equator on the rim; latitude = 90° (1 − r/c), longitude = atan2(row − c, col − c) + 90°. In stored arrays longitude increases clockwise; draw them with the row axis pointing up for the usual north-polar view.
- **Sign:** `sub` files hold the anomaly a (sublevel filtration, highs, "1-anticyclones", `A`); `sup` files hold −a (lows, "1-cyclones", `C`). Units are Pa in files, hPa in tables.
- **Persistence:** 8-connectivity (V-construction). Depth = value at the extremum − value at the saddle where its region merges with a deeper one.
- **Surrogate names:** `ft` Gaussian null, `iaaft` amplitude-adjusted null, `ftg`/`ftw` Gaussian surrogate of the whole record without seasonal windows, `hyb` 1948–78 structure with 1979–2023 mean and variance. Seeds are `1000 × first year + realisation`.

## Licence and citation

The code is released under the MIT licence (`LICENSE`). The reanalysis data are not part of this repository and remain under the terms of their providers (`data/README.md`). If you use the code, please cite the paper above.
