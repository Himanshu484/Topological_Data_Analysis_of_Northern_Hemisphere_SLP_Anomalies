# Persistence of the stored discs, and linking utilities

`pers_features.py` is the reference implementation of the paper's Proposition on duality (Section 2.5): the degree-one classes of the sublevel filtration of an image are its local extrema under 8-connectivity, and their persistence is the prominence of the extremum. It returns, for one disc, the extremum, the saddle and the elder-rule parent of every feature (parent −1 when the feature merges with the rim).

| File | Does |
|---|---|
| `pers_features.py` | Union–find persistence of one disc. |
| `extract_features.py` | Runs it on every day of 1948–2023 for both signs and compares each day with the stored Ripserer.jl diagrams and death vertices (55,480 of 55,480 diagrams identical). Writes `data/processed_data/prototype/feat_{sup,sub}_1948_2023.npz` (Pa). |
| `saddles_and_hysteresis.py` | Where the saddles sit (share of deep features that merge with the rim; correlation of depth with the value at the extremum, Section 3), and tracking with two thresholds. |
| `tracking.py` | Pixel ↔ (lat, lon), great-circle distance, nearest-neighbour linking; imported by the other analyses. |
| `run_all.sh` | `bash scripts/prototype/run_all.sh [Y0 Y1]`; text results in `output/prototype/`. About 10 minutes on 10 cores. |

Needs `data/processed_data/SLP_data_years/` and `representative_data/` (steps 2 of the main README). The folder name is historical.
