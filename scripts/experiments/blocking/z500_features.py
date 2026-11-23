"""
Persistence features (death = extremum, birth = saddle, parent) of Z500 polar discs,
built with the manuscript's projection (globe_visualization.reshape_to_polar_matrix)
and scripts/prototype/pers_features.py, for 1979-2023.

  za_sub  maxima of the Z500 anomaly     ("Z500 1-anticyclones")
  za_sup  minima of the Z500 anomaly     ("Z500 1-cyclones")
  zf_sub  maxima of the FULL Z500 field  (cut-off highs: any extratropical maximum of Z500
          is a closed 500 hPa anticyclone; its depth is its height above the col that joins
          it to the subtropical ridge)
  zf_sup  minima of the full Z500 field  (polar-vortex centres and cut-off lows)

Output: CACHE/feat_{kind}_1979_2023.npz in the prototype format, birth/death in gpm.

Engine: cripser.computePH(disc, maxdim=1), i.e. H1 of the V-construction sublevel filtration
of the padded disc (the manuscript's definition).  On 6 test discs (SLP sub/sup, Z500 anomaly
sub/sup, full Z500 sub/sup) its (birth, death) multiset and peak pixel are identical to
pers_features.py, ~10x faster.  It gives no parent, so parrow/parcol = -2 (unknown) and
srow/scol is the birth-edge pixel (within one pixel of pers_features' saddle).

    python scripts/experiments/blocking/z500_features.py [--jobs 1] [--kinds za_sub ...]
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np

from common import CACHE, Y0, EVAL0, EVAL1, disc
import cripser

KINDS = {'za_sub': ('z500_anom.npy', 'sub', 5.0), 'za_sup': ('z500_anom.npy', 'sup', 5.0),
         'zf_sub': ('z500_full.npy', 'sub', 1.0), 'zf_sup': ('z500_full.npy', 'sup', 1.0)}
W = 172


def one_year(args):
    y, kind = args
    fname, sub, keep = KINDS[kind]
    A = np.load(os.path.join(CACHE, fname), mmap_mode='r')
    rows = []
    for d in range(1, 366):
        g = disc(np.asarray(A[y - Y0, d - 1], np.float64), sub)
        P = cripser.computePH(g, maxdim=1)
        P = P[(P[:, 0] == 1) & (np.abs(P[:, 2] - P[:, 1]) >= keep)]
        for x in P:
            rows.append((y, d, x[1], x[2], int(x[6]), int(x[7]), int(x[3]), int(x[4]), -2, -2))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=2)
    ap.add_argument('--kinds', nargs='*', default=list(KINDS))
    a = ap.parse_args()
    for kind in a.kinds:
        out = os.path.join(CACHE, f'feat_{kind}_{EVAL0}_{EVAL1}.npz')
        if os.path.exists(out):
            continue
        if a.jobs > 1:
            with Pool(a.jobs) as P:
                res = P.map(one_year, [(y, kind) for y in range(EVAL0, EVAL1 + 1)])
        else:                                  # serial (the Pool stalled under memory pressure)
            res = []
            for y in range(EVAL0, EVAL1 + 1):
                res.append(one_year((y, kind)))
                print(f'  {kind} {y}', flush=True)
        rows = np.array([r for R in res for r in R], float)
        i16 = lambda k: rows[:, k].astype(np.int16)  # noqa: E731
        np.savez_compressed(out, y0=EVAL0, y1=EVAL1, year=i16(0), day=i16(1), birth=rows[:, 2],
                            death=rows[:, 3], prow=i16(4), pcol=i16(5), srow=i16(6), scol=i16(7),
                            parrow=i16(8), parcol=i16(9))
        print(f'{kind}: {len(rows)} features', flush=True)


if __name__ == '__main__':
    main()
