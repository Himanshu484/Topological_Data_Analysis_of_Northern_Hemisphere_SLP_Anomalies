# Test against a blocking index

Section 5.3 of the paper reports that the topological depth of highs does not detect blocking better than amplitude, and that hemispheric total persistence does not distinguish blocked from unblocked days. This folder is that test. Reference: a two-dimensional index of the Tibaldi–Molteni type computed from NCEP–NCAR R1 daily 500 hPa heights.

| File | Does |
|---|---|
| `download_z500.py` | R1 daily 500 hPa height, 0–90° N, 1948–2023, from NOAA PSL (about 8 MB per year). |
| `build_fields.py` | Z500 (full and anomaly) and SLP anomaly on the no-leap calendar. |
| `tm_index.py` | The reference blocking index, its climatology and events. |
| `z500_features.py` | Persistence features of the Z500 discs (`cripser`, V-construction). |
| `detectors.py` | Candidate detectors: tracked SLP highs by depth (T1) or amplitude, tracked Z500 anomaly highs by depth (T2) or amplitude (T2a), closed Z500 highs, grid-point baselines. |
| `run_eval.py`, `per_level.py` | Tuning on 1979–2000 and scores on 2001–2023 at sector-day, grid-day and event level (`eval_tuned.txt`, `eval_per_level.txt`; the paper quotes the event-level critical success index). |
| `composites.py` | Blocked against unblocked days for depth, nesting, local and hemispheric total persistence and Wasserstein distance (`composites.txt`; the paper quotes the area under the ROC curve). |

```bash
D=scripts/experiments/blocking
python $D/download_z500.py
python $D/build_fields.py
python $D/tm_index.py
python $D/z500_features.py --jobs 1 --kinds za_sub za_sup
python $D/z500_features.py --jobs 1 --kinds zf_sub zf_sup
python $D/run_eval.py
python $D/per_level.py
python $D/composites.py --k 3
```

Needs `data/processed_data/prototype/feat_*_1948_2023.npz` and the stored discs. Results: `output/experiments/blocking/`.
