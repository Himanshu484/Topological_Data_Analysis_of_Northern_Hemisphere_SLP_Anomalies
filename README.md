# Topological Data Analysis of Northern Hemisphere SLP Anomalies

Code and analysis pipeline for the paper:

> **Topological Data Analysis of Northern Hemisphere SLP Anomalies: Identifying and Tracking the Structural Skeleton of Atmospheric Pressure Systems**

## Method overview

1. **Field construction** — Daily sea-level pressure (SLP) anomalies over the Northern Hemisphere (1948–2023) are gridded as a 2D cubical complex.
2. **Filtration** — Sublevel-set filtration (on SLP, for low-pressure/cyclonic structures) and superlevel-set filtration (for high-pressure/anticyclonic structures) are computed on the complex.
3. **Persistent homology** — 1-dimensional persistent homology (1-holes) is computed on each filtration using [`ripser`](https://github.com/scikit-tda/ripser.py) / [`tcripser`](https://github.com/shizuo-kaji/CubicalRipser_3dim), yielding a persistence diagram per day. Each surviving 1-cycle is a **1-cyclone** (sub-level) or **1-anticyclone** (super-level), with its persistence (birth/death pressure) quantifying structural depth/intensity, and its representative cycle giving a spatial footprint used to estimate area.
4. **Tracking** — Consecutive daily persistence diagrams are matched using the 2-Wasserstein distance (`persim.wasserstein`) between persistence pairs, filtered by geodesic distance between representative-cycle centroids, to build day-to-day feature trajectories.
5. **Intercomparison** — TDA-derived cyclone tracks are benchmarked against a classical geometric tracker (a simplified Murray-Simmonds 1991 detection/tracking scheme) on the same SLP data, using the Hit Rate / False Alarm Rate / Critical Success Index framework of Bourdin et al. (2022).
6. **Case studies** — Event-window analyses isolate the persistence dynamics around the August 2003 European heatwave (blocking high) and the early-2012 cold spell.

## Repository structure

```
.
├── data/
│   ├── raw/                    # Input datasets (not tracked in git — see "Data" below)
│   └── processed_data/         # Derived arrays/diagrams (not tracked in git; regenerated from notebooks)
│       ├── SLP_data_years/     # Per-day SLP anomaly grids, by year
│       ├── persistence_data/   # Persistence diagrams (sub/sup-level)
│       ├── representative_data/# Representative cycles (spatial footprints) per feature
│       ├── area_data/          # Feature area time series
│       ├── feature_tracking/   # Wasserstein-matched trajectories
│       └── ms_tracker/         # Murray-Simmonds comparison tracker output
├── src/                        # Core Python modules imported by the notebooks
│   ├── cubical_pers_and_filt_visual.py  # Cubical persistence diagram plotting helpers
│   ├── globe_visualization.py           # Map/globe plotting of SLP fields and features
│   ├── feature_tracking.py              # Persistence loading, area calc, Wasserstein tracking
│   └── comparison_tracker.py            # Murray-Simmonds (1991) tracker + Bourdin et al. (2022) metrics
├── notebook/                   # End-to-end analysis pipeline (see "Usage" below)
├── julia/                      # Ripserer.jl cross-check of representative cycles
├── output/
│   ├── figures/                # Generated plots (not tracked in git)
│   └── trajectories/           # Exported trajectory data (not tracked in git)
├── requirements.txt
└── .gitignore
```

## Data

The analysis uses:

- **NCEP/NCAR Reanalysis 1 daily sea-level pressure** (`slp.daily.nc`), 1948–2023 — the primary field, obtained from the [NOAA Physical Sciences Laboratory](https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html). This file is ~575 MB and is **not included in this repository**; download it from PSL and place it at `data/raw/slp.daily.nc`.
- Auxiliary daily/monthly climate indices (AMO, ENSO, NAO, PDO, EA, SCAND, AO) used for correlation analysis, stored as small `.mat`/`.data`/`.nc` files under `data/raw/`.

All of `data/` and `output/` are excluded from version control via `.gitignore` because the processed intermediate data (per-day persistence diagrams, representative cycles, tracked trajectories) totals several gigabytes and is fully reproducible from the raw SLP file by running the notebooks in order below.

## Installation

Requires **Python 3.11**. `requirements.txt` is pinned to the versions used in the `climateenv` conda environment this project was developed with.

```bash
conda create -n climateenv python=3.11
conda activate climateenv
pip install -r requirements.txt
```

`cartopy` and `netCDF4` depend on GEOS/PROJ/HDF5 system libraries. If they fail to build from the `pip install` above, install those two via conda-forge first, then re-run `pip install -r requirements.txt` for the rest:

```bash
conda install -c conda-forge cartopy netcdf4
```

A plain `venv` (without conda) also works as long as GEOS/PROJ/HDF5 are already available on your system:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Note: `cripser` also provides the `tcripser` module imported by several notebooks — do not install a separate `tcripser` package. `notebook/08_video_frames.ipynb` additionally needs `opencv-python` (`cv2`), which is not part of `climateenv`; install it separately only if you need that notebook.

The `julia/` scripts additionally require [Julia](https://julialang.org/) with the `NPZ`, `Plots`, `Ripserer`, and `JSON` packages.

## Usage

Run the notebooks in `notebook/` in order; each stage reads the outputs of the previous one from `data/processed_data/`:

| Notebook | Purpose |
|---|---|
| `01_data_visualization_and_analysis.ipynb` | Load and inspect the raw SLP field |
| `02_data_preprocessing_and_saving.ipynb` | Compute daily anomalies, save per-day/per-year grids |
| `03_persistent_analysis.ipynb` | Compute sub/sup-level cubical persistence diagrams; correlate with climate indices |
| `04_storm_area_analysis.ipynb` | Derive spatial area of tracked features from representative cycles |
| `05_storm_tracking_visualization.ipynb` | Visualize Wasserstein-matched feature trajectories |
| `06_storm_tracking_analysis.ipynb` | Statistical analysis of tracked trajectories |
| `07_ms_comparison.ipynb` | Intercompare TDA tracker vs. Murray-Simmonds (1991) tracker (Bourdin et al. 2022 metrics) |
| `08_video_frames.ipynb` | Render frame sequences for animations |
| `09_event_window_persistence.ipynb` | Case study: February 2012 cold spell |
| `10_event_window_persistence_August_2003.ipynb` | Case study: August 2003 European heatwave |

The `julia/trajectory_representative_*.jl` scripts representative-cycle extraction using `Ripserer.jl`.

## Citation

If you use this code, please cite the associated paper:

```bibtex
@article{yadav_slp_tda,
  title   = {Topological Data Analysis of Northern Hemisphere SLP Anomalies: Identifying and Tracking the Structural Skeleton of Atmospheric Pressure Systems},
  author  = {Yadav, Himanshu},
  journal = {},
  year    = {}
}
```
