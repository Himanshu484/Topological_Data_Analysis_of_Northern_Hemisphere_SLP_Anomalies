"""
Sweep every detector over its threshold, tune on 1979-2000 (max sector-day CSI against the
2-D TM reference, all sectors and seasons pooled), and score on 2001-2023.

Outputs (OUT):
  eval_sweep.csv        every detector x threshold x period: all scores
  eval_tuned.txt        tuned thresholds, test-period scores, by sector and season,
                        bootstrap CIs of CSI differences
  eval_performance.png  performance diagrams (test period)
Cache: CACHE/det_{name}.npz  snapshots of each tuned detector (for composites / cases)

    python scripts/experiments/blocking/run_eval.py
"""
import os
import time

import numpy as np
import pandas as pd

from common import CACHE, OUT, PROTO_FEAT, LAT, Y0, EVAL0, EVAL1, TRAIN, TEST, SEASONS, MONTH
import detectors as dt

TRY = list(range(TRAIN[0], TRAIN[1] + 1))
TEY = list(range(TEST[0], TEST[1] + 1))
SECS = dt.SECS


def monthly_scale(A):
    """Per-slot normaliser: SD of the anomaly over 45-80N in that calendar month / annual SD."""
    rows = dt.ROWS_DET
    sd_all = A[:, rows][::3].std()
    s = np.zeros(13)
    for m in range(1, 13):
        sel = MONTH[dt.SLOT - 1] == m
        s[m] = A[sel][:, rows][::2].std() / sd_all
    return s[MONTH[dt.SLOT - 1]], s


def main():
    t0 = time.time()
    ref = dt.ref_tm()
    A_z = np.asarray(np.load(os.path.join(CACHE, 'z500_anom.npy'), mmap_mode='r')[EVAL0 - Y0:EVAL1 - Y0 + 1]
                     ).reshape(dt.NT, 37, 144)
    A_s = np.asarray(np.load(os.path.join(CACHE, 'slp_anom.npy'), mmap_mode='r')[EVAL0 - Y0:EVAL1 - Y0 + 1]
                     ).reshape(dt.NT, 37, 144) / 100.0
    sc_z, mz = monthly_scale(A_z)
    sc_s, ms = monthly_scale(A_s)
    print('monthly SD scale Z500:', np.round(mz[1:], 2), '\n                 SLP :', np.round(ms[1:], 2), flush=True)

    # --- second reference (DG-type) and the TM index as a 'detector' (inter-index agreement)
    det_dg, ref_dg, thr_dg = dt.dg_index(A_z, return_points=True)
    print(f'DG index: {det_dg["eid"].max() + 1} events; monthly thresholds (gpm)', np.round(thr_dg[1:]),
          f'[{time.time() - t0:.0f}s]', flush=True)
    E = np.load(os.path.join(CACHE, 'tm_events.npz'), allow_pickle=True)
    k = (E['d_t'] >= dt.T0) & (E['d_t'] < dt.T0 + dt.NT)
    det_tm = dict(t=E['d_t'][k] - dt.T0, lat=E['d_lat'][k], lon=E['d_lon'][k], eid=E['d_eid'][k])

    # --- detector registry: name -> (family, thresholds, callable)
    S = os.path.join(PROTO_FEAT, 'feat_sub_1948_2023.npz')
    ZA = os.path.join(CACHE, f'feat_za_sub_{EVAL0}_{EVAL1}.npz')
    ZL = os.path.join(CACHE, f'feat_za_sup_{EVAL0}_{EVAL1}.npz')
    ZF = os.path.join(CACHE, f'feat_zf_sub_{EVAL0}_{EVAL1}.npz')
    tr_s = dt.TopoTracks(S, 3.0, 100.0)
    print(f'SLP highs: {len(tr_s.F["slot"])} features >= 3 hPa, {tr_s.ntracks} tracks [{time.time() - t0:.0f}s]',
          flush=True)
    have_z = os.path.exists(ZA) and os.path.exists(ZL) and os.path.exists(ZF)
    reg = {}
    th_s = [3, 4, 5, 6, 8, 10, 12, 14, 16, 20]
    th_sn = [0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 2.0]       # multiples of 10 hPa x monthly scale
    reg['T1 SLP depth'] = ('topo', th_s, lambda x: tr_s.detect(x))
    reg['T1a SLP amplitude'] = ('ablation', th_s, lambda x: tr_s.detect(x, use='amp'))
    reg['Te1 SLP depth, Eulerian r=1500'] = ('topo', th_s, lambda x: dt.topo_loc(tr_s.F, x, r=1500.0))
    reg['B1 SLP gridpoint'] = ('baseline', th_s, lambda x: dt.grid_detector(A_s, x))
    reg['T1n SLP depth (seas-norm)'] = ('topo', th_sn, lambda x: tr_s.detect(10 * x, scale=sc_s))
    reg['B1n SLP gridpoint (seas-norm)'] = ('baseline', th_sn, lambda x: dt.grid_detector(A_s, 10 * x, scale=sc_s))
    if have_z:
        tr_z = dt.TopoTracks(ZA, 20.0, 1.0)
        tr_f = dt.TopoTracks(ZF, 2.0, 1.0)
        lows = dt.load_feats(ZL, 20.0, 1.0, EVAL0, EVAL1)
        print(f'Z500a highs: {len(tr_z.F["slot"])} feats, {tr_z.ntracks} tracks; Z500 cut-off highs: '
              f'{len(tr_f.F["slot"])} feats, {tr_f.ntracks} tracks [{time.time() - t0:.0f}s]', flush=True)
        dip = {tl: dt.dipole_mask(tr_z.F, lows, tl) for tl in (40, 80)}
        th_z = [30, 40, 50, 60, 80, 100, 120, 140, 160, 200]
        th_zn = [0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 2.0]    # multiples of 100 gpm x monthly scale
        th_f = [2, 3, 5, 10, 15, 20, 30, 40, 60, 80]
        reg['T2 Z500a depth'] = ('topo', th_z, lambda x: tr_z.detect(x))
        reg['T2a Z500a amplitude'] = ('ablation', th_z, lambda x: tr_z.detect(x, use='amp'))
        reg['Te2 Z500a depth, Eulerian r=1500'] = ('topo', th_z, lambda x: dt.topo_loc(tr_z.F, x, r=1500.0))
        reg['B2 Z500a gridpoint'] = ('baseline', th_z, lambda x: dt.grid_detector(A_z, x))
        reg['T2n Z500a depth (seas-norm)'] = ('topo', th_zn, lambda x: tr_z.detect(100 * x, scale=sc_z))
        reg['B2n Z500a gridpoint (seas-norm)'] = ('baseline', th_zn,
                                                 lambda x: dt.grid_detector(A_z, 100 * x, scale=sc_z))
        reg['T2d40 Z500a depth + dipole(40)'] = ('topo', th_z, lambda x: tr_z.detect(x, extra=dip[40]))
        reg['T2d80 Z500a depth + dipole(80)'] = ('topo', th_z, lambda x: tr_z.detect(x, extra=dip[80]))
        reg['T3 Z500 cut-off high prominence'] = ('topo', th_f, lambda x: tr_f.detect(x))
        reg['Te3 Z500 cut-off high, Eulerian r=1500'] = ('topo', th_f, lambda x: dt.topo_loc(tr_f.F, x, r=1500.0))
    reg['R2 DG anomaly-overlap index'] = ('index', [0.90], lambda x: det_dg)

    rows = []
    C_all = {}
    for name, (fam, ths, fn) in reg.items():
        for th in ths:
            det = fn(th)
            C = dt.counts(det, ref)
            C_all[(name, th)] = C
            for per, ys in (('train', TRY), ('test', TEY)):
                s = dt.scores(C, ys)
                rows.append(dict(detector=name, family=fam, thr=th, period=per, n_snap=len(det['t']),
                                 **s))
        print(f'  {name}: done [{time.time() - t0:.0f}s]', flush=True)
    sw = pd.DataFrame(rows)
    sw.to_csv(os.path.join(OUT, 'eval_sweep.csv'), index=False)

    # --- tuning and test-period tables
    L = []
    tuned = {}
    for name in reg:
        tr = sw[(sw.detector == name) & (sw.period == 'train')]
        best = tr.loc[tr.sd_csi.idxmax(), 'thr']
        tuned[name] = best
    L.append(f'Reference: 2-D TM (Davini 2012) blocking events, phi0 >= 40N. Tuning 1979-2000 on sector-day '
             f'CSI; scores 2001-2023.\n')
    L.append(f'{"detector":42} {"thr":>6} | {"sd POD":>6} {"FAR":>5} {"CSI":>5} {"bias":>5} | '
             f'{"gd POD":>6} {"FAR":>5} {"CSI":>5} | {"ev POD":>6} {"FAR":>5} {"CSI":>5} {"n_det":>6}')
    for name in reg:
        s = dt.scores(C_all[(name, tuned[name])], TEY)
        L.append(f'{name:42} {tuned[name]:6g} | {s["sd_pod"]:6.3f} {s["sd_far"]:5.3f} {s["sd_csi"]:5.3f} '
                 f'{s["sd_bias"]:5.2f} | {s["gd_pod"]:6.3f} {s["gd_far"]:5.3f} {s["gd_csi"]:5.3f} | '
                 f'{s["ev_pod"]:6.3f} {s["ev_far"]:5.3f} {s["ev_csi"]:5.3f} {s["ev_ndet"]:6.0f}')
    C_dgtm = dt.counts(det_tm, ref_dg)
    s = dt.scores(C_dgtm, TEY)
    L.append(f'{"[TM index scored against DG reference]":42} {"":6} | {s["sd_pod"]:6.3f} {s["sd_far"]:5.3f} '
             f'{s["sd_csi"]:5.3f} {s["sd_bias"]:5.2f} | {s["gd_pod"]:6.3f} {s["gd_far"]:5.3f} {s["gd_csi"]:5.3f} | '
             f'{s["ev_pod"]:6.3f} {s["ev_far"]:5.3f} {s["ev_csi"]:5.3f} {s["ev_ndet"]:6.0f}')
    L.append(f'(reference events in test period: {s["ev_nref"]:.0f} DG; '
             f'{dt.scores(C_all[(list(reg)[0], tuned[list(reg)[0]])], TEY)["ev_nref"]:.0f} TM)')

    # by sector and season, sector-day CSI
    L.append('\nSector-day CSI by sector (test period):')
    L.append(f'{"detector":42} ' + ' '.join(f'{s:>6}' for s in SECS))
    for name in reg:
        C = C_all[(name, tuned[name])]
        L.append(f'{name:42} ' + ' '.join(f'{dt.scores(C, TEY, sec=k)["sd_csi"]:6.3f}' for k in range(4)))
    L.append('\nSector-day CSI by season (test period):')
    L.append(f'{"detector":42} ' + ' '.join(f'{s:>6}' for s in SEASONS))
    for name in reg:
        C = C_all[(name, tuned[name])]
        L.append(f'{name:42} ' + ' '.join(f'{dt.scores(C, TEY, seas=s)["sd_csi"]:6.3f}' for s in SEASONS))
    L.append('\nEvent CSI by sector (test period):')
    L.append(f'{"detector":42} ' + ' '.join(f'{s:>6}' for s in SECS))
    for name in reg:
        C = C_all[(name, tuned[name])]
        L.append(f'{name:42} ' + ' '.join(f'{dt.scores(C, TEY, sec=k)["ev_csi"]:6.3f}' for k in range(4)))
    L.append('\nEvent CSI by onset season (test period):')
    L.append(f'{"detector":42} ' + ' '.join(f'{s:>6}' for s in SEASONS))
    for name in reg:
        C = C_all[(name, tuned[name])]
        L.append(f'{name:42} ' + ' '.join(f'{dt.scores(C, TEY, seas=s)["ev_csi"]:6.3f}' for s in SEASONS))

    # bootstrap: topological vs its non-topological counterpart
    pairs = [('T1 SLP depth', 'B1 SLP gridpoint'), ('T1 SLP depth', 'T1a SLP amplitude'),
             ('T1n SLP depth (seas-norm)', 'B1n SLP gridpoint (seas-norm)')]
    if have_z:
        pairs += [('T2 Z500a depth', 'B2 Z500a gridpoint'), ('T2 Z500a depth', 'T2a Z500a amplitude'),
                  ('T2n Z500a depth (seas-norm)', 'B2n Z500a gridpoint (seas-norm)'),
                  ('T3 Z500 cut-off high prominence', 'B2 Z500a gridpoint'),
                  ('T3 Z500 cut-off high prominence', 'T2 Z500a depth'),
                  ('T2d40 Z500a depth + dipole(40)', 'T2 Z500a depth'),
                  ('Te2 Z500a depth, Eulerian r=1500', 'B2 Z500a gridpoint'),
                  ('Te1 SLP depth, Eulerian r=1500', 'B1 SLP gridpoint'),
                  ('T3 Z500 cut-off high prominence', 'R2 DG anomaly-overlap index')]
    L.append('\nCSI difference A - B, test period, year-block bootstrap 95% CI (1000 draws):')
    for a, b in pairs:
        parts = []
        for key in ('sd_csi', 'gd_csi', 'ev_csi'):
            d, lo, hi = dt.boot_diff(C_all[(a, tuned[a])], C_all[(b, tuned[b])], TEY, key)
            parts.append(f'{key[:2]} {d:+.3f} [{lo:+.3f}, {hi:+.3f}]')
        L.append(f'  {a}  vs  {b}:  ' + ';  '.join(parts))
    txt = '\n'.join(L)
    print(txt)
    open(os.path.join(OUT, 'eval_tuned.txt'), 'w').write(txt + '\n')

    # --- cache tuned detectors
    for name in reg:
        fam, ths, fn = reg[name]
        det = fn(tuned[name])
        key = name.split()[0]
        np.savez(os.path.join(CACHE, f'det_{key}.npz'), thr=tuned[name],
                 **{k: v for k, v in det.items()})
    np.savez(os.path.join(CACHE, 'ref_dg.npz'), **ref_dg)

    # --- performance diagram
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    sr_g, pod_g = np.meshgrid(np.linspace(0.01, 1, 100), np.linspace(0.01, 1, 100))
    csi_g = 1 / (1 / sr_g + 1 / pod_g - 1)
    te = sw[sw.period == 'test']
    for ax, lvl, ttl in zip(axes, ['sd', 'gd', 'ev'], ['sector-day', 'grid-day (1000 km)', 'event']):
        cs = ax.contour(sr_g, pod_g, csi_g, levels=np.arange(0.1, 1, 0.1), colors='0.8', linewidths=0.7)
        ax.clabel(cs, fmt='%.1f', fontsize=7)
        for name in reg:
            d = te[te.detector == name]
            fam = reg[name][0]
            ls = {'topo': '-', 'ablation': ':', 'baseline': '--', 'index': 'none'}[fam]
            mk = {'topo': 'o', 'ablation': 's', 'baseline': '^', 'index': '*'}[fam]
            ax.plot(1 - d[f'{lvl}_far'], d[f'{lvl}_pod'], ls=ls, marker=mk, ms=4 if fam != 'index' else 12,
                    label=name)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('success ratio (1 - FAR)')
        ax.set_ylabel('POD')
        ax.set_title(f'{ttl}, 2001-2023 vs 2-D TM')
    axes[-1].legend(fontsize=6.5, loc='upper right')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'eval_performance.png'), dpi=110)
    print(f'done [{time.time() - t0:.0f}s]')


if __name__ == '__main__':
    main()
