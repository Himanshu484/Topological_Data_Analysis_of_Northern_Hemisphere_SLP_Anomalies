"""
Fairer comparison: every detector tuned SEPARATELY for each verification level (sector-day,
grid-day, event CSI) on 1979-2000, scored on 2001-2023, with year-block bootstrap CIs of the
key differences.  Reads eval_sweep.csv from run_eval.py.

Output: OUT/eval_per_level.txt

    python scripts/experiments/blocking/per_level.py
"""
import os

import numpy as np
import pandas as pd

from common import CACHE, OUT, PROTO_FEAT, Y0, EVAL0, EVAL1, TEST
import detectors as dt

TEY = list(range(TEST[0], TEST[1] + 1))


def main():
    sw = pd.read_csv(os.path.join(OUT, 'eval_sweep.csv'))
    tr, te = sw[sw.period == 'train'], sw[sw.period == 'test']
    L = ['Each detector tuned separately per level on 1979-2000 (max CSI of that level); test 2001-2023 CSI.',
         f'{"detector":42} {"sd thr":>6} {"sd CSI":>6} | {"gd thr":>6} {"gd CSI":>6} | {"ev thr":>6} {"ev CSI":>6} '
         f'{"ev POD":>6} {"ev FAR":>6}']
    best = {}
    for d in sw.detector.unique():
        row = [f'{d:42}']
        for lvl in ('sd', 'gd', 'ev'):
            a = tr[tr.detector == d]
            th = a.loc[a[f'{lvl}_csi'].idxmax(), 'thr']
            b = te[(te.detector == d) & (te.thr == th)].iloc[0]
            best[(d, lvl)] = th
            row.append(f'{th:6g} {b[f"{lvl}_csi"]:6.3f}')
            if lvl == 'ev':
                row[-1] += f' {b.ev_pod:6.3f} {b.ev_far:6.3f}'
        L.append(' | '.join(row))

    # bootstrap of the key contrasts at their per-level optimum
    ref = dt.ref_tm()
    A_z = np.asarray(np.load(os.path.join(CACHE, 'z500_anom.npy'), mmap_mode='r')[EVAL0 - Y0:EVAL1 - Y0 + 1]
                     ).reshape(dt.NT, 37, 144)
    tr_z = dt.TopoTracks(os.path.join(CACHE, f'feat_za_sub_{EVAL0}_{EVAL1}.npz'), 20.0, 1.0)
    tr_f = dt.TopoTracks(os.path.join(CACHE, f'feat_zf_sub_{EVAL0}_{EVAL1}.npz'), 2.0, 1.0)
    fn = {'T2 Z500a depth': lambda x: tr_z.detect(x),
          'T2a Z500a amplitude': lambda x: tr_z.detect(x, use='amp'),
          'B2 Z500a gridpoint': lambda x: dt.grid_detector(A_z, x),
          'T3 Z500 cut-off high prominence': lambda x: tr_f.detect(x),
          'R2 DG anomaly-overlap index': None}
    det_dg = dt.dg_index(A_z)
    C = {}
    for name, f in fn.items():
        for lvl in ('gd', 'ev'):
            th = best[(name, lvl)]
            key = (name, th)
            if key not in C:
                C[key] = dt.counts(det_dg if f is None else f(th), ref)
    pairs = [('T2 Z500a depth', 'T2a Z500a amplitude'), ('T2 Z500a depth', 'B2 Z500a gridpoint'),
             ('T2a Z500a amplitude', 'B2 Z500a gridpoint'), ('T3 Z500 cut-off high prominence', 'B2 Z500a gridpoint'),
             ('T3 Z500 cut-off high prominence', 'T2a Z500a amplitude'),
             ('T3 Z500 cut-off high prominence', 'R2 DG anomaly-overlap index')]
    L.append('\nCSI difference A - B at per-level optima, test period, year-block bootstrap 95% CI:')
    for a, b in pairs:
        parts = []
        for lvl in ('gd', 'ev'):
            d, lo, hi = dt.boot_diff(C[(a, best[(a, lvl)])], C[(b, best[(b, lvl)])], TEY, f'{lvl}_csi')
            parts.append(f'{lvl} {d:+.3f} [{lo:+.3f}, {hi:+.3f}]')
        L.append(f'  {a} vs {b}: ' + '; '.join(parts))
    txt = '\n'.join(L)
    print(txt)
    open(os.path.join(OUT, 'eval_per_level.txt'), 'w').write(txt + '\n')


if __name__ == '__main__':
    main()
