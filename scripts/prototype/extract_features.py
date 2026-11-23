"""
Extract every H1 feature with its death cell, birth cell (saddle) and parent,
for both filtrations, and validate against the stored Ripserer output.

Validation, per day:
  * the (birth, death) multiset equals the stored JSON diagram
  * for depth >= 5 hPa, the peak pixel equals the stored death vertex
    (the vertex of the death cube carrying the death value)

Output: data/processed_data/prototype/feat_{sup|sub}_{y0}_{y1}.npz (birth/death in Pa)

    python scripts/prototype/extract_features.py [--years 1948 2023] [--jobs 10]
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from pers_features import features  # noqa: E402

OUT = os.path.join(ROOT, 'data', 'processed_data', 'prototype')
W = 172


def one_year(args):
    y, t = args
    rows = []
    bad_days = px_ok = px_bad = 0
    for d in range(1, 366):
        g = np.load(os.path.join(ROOT, 'data', 'processed_data', 'SLP_data_years', str(y),
                                 f'slp_{t}_{y}_day_{d}.npy'))
        F = features(g, g == g[0, 0])
        with open(os.path.join(ROOT, 'data', 'processed_data', 'representative_data', str(y),
                               f'slp_{t}_{y}_day_{d}.json')) as f:
            J = json.load(f)
        ref = sorted(zip(np.round(J['births'], 2), np.round(J['deaths'], 2)))
        mine = sorted((round(s, 2), round(pk, 2)) for pk, s, _, _, _ in F)
        bad_days += ref != mine
        stored = {}
        for b, de, vs in zip(J['births'], J['deaths'], J['death_simplex_vertices']):
            if de - b >= 500:
                hit = [tuple(v) for v in vs if abs(float(g[v[0], v[1]]) - de) <= 1e-6 * max(1, abs(de))]
                if len(hit) == 1:
                    stored[(round(b, 2), round(de, 2))] = hit[0]
        for pk, s, ip, isd, ipar in F:
            key = (round(s, 2), round(pk, 2))
            if key in stored:
                if stored[key] == divmod(ip, W):
                    px_ok += 1
                else:
                    px_bad += 1
            pr, pc = divmod(ipar, W) if ipar >= 0 else (-1, -1)
            rows.append((y, d, s, pk, *divmod(ip, W), *divmod(isd, W), pr, pc))
    return rows, bad_days, px_ok, px_bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', nargs=2, type=int, default=[1948, 2023])
    ap.add_argument('--jobs', type=int, default=10)
    a = ap.parse_args()
    y0, y1 = a.years
    os.makedirs(OUT, exist_ok=True)
    for t in ['sup', 'sub']:
        with Pool(a.jobs) as P:
            res = P.map(one_year, [(y, t) for y in range(y0, y1 + 1)])
        rows = np.array([r for R in res for r in R[0]], float)
        print(f'{t}: {len(rows)} features | days whose diagram differs from the stored one: '
              f'{sum(R[1] for R in res)} of {365 * (y1 - y0 + 1)} | death pixel matches '
              f'{sum(R[2] for R in res)}, mismatches {sum(R[3] for R in res)}', flush=True)
        i16 = lambda k: rows[:, k].astype(np.int16)
        np.savez_compressed(os.path.join(OUT, f'feat_{t}_{y0}_{y1}.npz'), y0=y0, y1=y1,
                            year=i16(0), day=i16(1), birth=rows[:, 2], death=rows[:, 3],
                            prow=i16(4), pcol=i16(5),
                            srow=i16(6), scol=i16(7), parrow=i16(8), parcol=i16(9))


if __name__ == '__main__':
    main()
