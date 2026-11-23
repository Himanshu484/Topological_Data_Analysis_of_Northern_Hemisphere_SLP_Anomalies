"""
What the birth cell (saddle) carries, and what two-level thresholds do.

  * where saddles sit relative to the extremum, and how far they move per day
  * how often the parent is the disc rim, and how close saddle values are to 0,
    i.e. how often topological depth is just the anomaly amplitude
  * nearest-neighbour tracking with one threshold (10 hPa) against hysteresis
    tracking (follow features down to 5 or 3 hPa, keep tracks that reach 10 hPa)

    python scripts/prototype/saddles_and_hysteresis.py [--years 1948 2023]
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from tracking import load, run, summary, tracks, RE_KM  # noqa: E402

DATA = os.path.join(ROOT, 'data', 'processed_data', 'prototype')


def gc(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    return 2 * RE_KM * np.arcsin(np.sqrt(np.sin((la1 - la2) / 2) ** 2
                                         + np.cos(la1) * np.cos(la2) * np.sin((lo1 - lo2) / 2) ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', nargs=2, type=int, default=[1948, 2023])
    y0, y1 = ap.parse_args().years
    for t, name in [('sup', '1-cyclones'), ('sub', '1-anticyclones')]:
        path = os.path.join(DATA, f'feat_{t}_{y0}_{y1}.npz')
        F = load(path, 10.0)
        r = run(F, 'nn')
        i = np.where(r['succ'] >= 0)[0]
        j = r['succ'][i]
        move_pk = gc(F['plat'][i], F['plon'][i], F['plat'][j], F['plon'][j])
        move_sd = gc(F['slat'][i], F['slon'][i], F['slat'][j], F['slon'][j])
        sep = gc(F['plat'], F['plon'], F['slat'], F['slon'])
        b, d = F['birth'], F['death']
        print(f'\n== {name} {y0}-{y1}, depth >= 10 hPa: {len(b)} features, {len(i)} nearest-neighbour links')
        print(f'   extremum-to-saddle distance: median {np.median(sep):.0f} km')
        print(f'   daily move of the extremum: median {np.median(move_pk):.0f} km (90th pct {np.percentile(move_pk, 90):.0f})')
        print(f'   daily move of the saddle:   median {np.median(move_sd):.0f} km (90th pct {np.percentile(move_sd, 90):.0f}); '
              f'> 2000 km in {100 * np.mean(move_sd > 2000):.1f}% of links')
        print(f'   parent is the disc rim: {100 * np.mean(F["parrow"] < 0):.1f}%; |saddle value| < 2 hPa: '
              f'{100 * np.mean(np.abs(b) < 2):.1f}%; median depth/|extremum| {np.median((d - b) / np.abs(d)):.2f}; '
              f'corr(depth, |extremum|) {np.corrcoef(d - b, np.abs(d))[0, 1]:.3f}')
        s10 = summary(F, r)
        n10_multi = sum(len(m) for m in tracks(r) if len(m) >= 2)
        print(f'   one threshold (10 hPa):  tracks {s10["N"]}, 1-day {s10["pct1"]:.1f}%, mean {s10["meanL"]:.2f} d, '
              f'>= 5 d {s10["n_ge5"]}, max {s10["maxL"]} d; >= 10 hPa snapshots in multi-day tracks '
              f'{100 * n10_multi / len(F["slot"]):.1f}%')
        for lo in [5.0, 3.0]:
            H = load(path, lo)
            rh = run(H, 'nn')
            keep = H['depth'] >= 10
            sh = summary(H, rh, keep_mask=keep)
            T = [m for m in tracks(rh) if keep[m].any()]
            n10m = sum(int(keep[m].sum()) for m in T if len(m) >= 2)
            print(f'   hysteresis {lo:g}/10 hPa:   tracks {sh["N"]}, 1-day {sh["pct1"]:.1f}%, mean {sh["meanL"]:.2f} d, '
                  f'>= 5 d {sh["n_ge5"]}, max {sh["maxL"]} d; >= 10 hPa snapshots in multi-day tracks '
                  f'{100 * n10m / int(keep.sum()):.1f}%', flush=True)


if __name__ == '__main__':
    main()
