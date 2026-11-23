# Topological depth against other detection criteria

Section 4.6 of the paper (Table 14, and the statistics of how trajectories end). Five criteria are applied to the same extrema of the R1 anomaly, 1948–2023, each thresholded month by month to the number of detections that topological depth gives at 10 hPa, and linked by nearest-neighbour assignment within 1000 km:

- **P** topological depth (prominence);
- **A** amplitude of the anomaly at the extremum;
- **AN** amplitude, keeping only extrema with no stronger extremum within 1000 km;
- **L** depth relative to the mean over an annulus of 500–1000 km;
- **C** closed-contour depth with the contour limited to 1200 km.

| File | Does |
|---|---|
| `descriptors.py` | One pass over the stored discs: ring means, closed-contour depth and neighbourhood values for every extremum. |
| `common.py` | Per-extremum table (parent, orography, land mask), detector scores, density calibration, linking, chance baseline from fields rotated in longitude. |
| `detectors.py`, `detectors_monthly.py` | The comparison at matched density, over the year and month by month (Table 14 uses the monthly version). |
| `lifecycle.py` | How trajectories end: the share that end with the feature still present as a strong extremum but no longer prominent (absorption into the parent), against the chance rate. |
| `run_all.sh` | The whole chain, about 40 minutes on 2 cores. |

Needs `data/processed_data/prototype/feat_*_1948_2023.npz`, the stored discs, and the R1 orography and land mask (`data/README.md`). Results: `output/experiments/beyond_amplitude/*.json` and `*.out`.
