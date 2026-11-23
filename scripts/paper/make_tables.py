"""
Tables of the manuscript, written as LaTeX fragments to <manuscript>/tables/
(see paths.py).

    python scripts/paper/make_null_tables.py        # the null-field fragments (run first)
    python scripts/paper/make_tables.py

The manuscript uses the names "1-cyclones" and "1-anticyclones" and
tracks with the lifted Wasserstein matcher (src/lifted_tracker.py), so
  * tables that involve no tracking are copied from output/paper/null_tables/ with the
    feature names replaced;
  * every track statistic is taken from output/paper/analysis_{R1,ERA5}.json
    (scripts/paper/trajectory_analysis.py).
numbers_hybrid.json holds every value used in the text.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_null_tables as T  # noqa: E402
import paths  # noqa: E402

ROOT = T.ROOT
TAB = paths.TABLES
A = {s: json.load(open(os.path.join(ROOT, 'output', 'paper', f'analysis_{s}.json'))) for s in ('R1', 'ERA5')}
NAME = {'C': '1-cyclones', 'A': '1-anticyclones'}
NUM = {}


def write(name, body):
    open(os.path.join(TAB, name), 'w').write(body)
    print('wrote', name)


def renamed_copies():
    for f in ('tab_asymmetry', 'tab_seasons', 'tab_diagnostics', 'tab_coherence', 'tab_footprint_sensitivity', 'tab_step'):
        s = open(os.path.join(T.TAB, f + '.tex')).read()
        for a, b in (('{Lows}', '{1-cyclones}'), ('{Highs}', '{1-anticyclones}'), ('Lows &', '1-cyclones &'),
                     ('Highs &', '1-anticyclones &'), (', lows', ', 1-cyclones'), (', highs', ', 1-anticyclones'),
                     ('Low/high ratio of', 'Ratio 1-cyclones/1-anticyclones of')):
            s = s.replace(a, b)
        assert 'ows' not in s.replace('1-cyclones', '') and 'ighs' not in s, f
        write(f + '.tex', s)


# ------------------------------------------------------------------ trajectory statistics
def tracks_table():
    def ci(src, c, k, fmt):
        lo, hi = A[src][c]['ci95'][k]
        return f'{fmt.format(A[src][c]["reference"][KEY[k]] if k in KEY else 0)} [{fmt.format(lo)}, {fmt.format(hi)}]'
    KEY = {'tracks_per_year': 'tracks_per_year', 'mean_length': 'mean_length', 'pct_1d': 'pct_1d', 'pct_ge5d': 'pct_ge5d',
           'pct_ge10d': 'pct_ge10d'}
    rows = [('Number of trajectories', lambda s, c: f"{A[s][c]['reference']['n_tracks']:,}".replace(',', '{,}')),
            ('\\quad per year', lambda s, c: ci(s, c, 'tracks_per_year', '{:.0f}')),
            ('Mean length (d)', lambda s, c: ci(s, c, 'mean_length', '{:.2f}')),
            ('Lasting 1 d (\\%)', lambda s, c: ci(s, c, 'pct_1d', '{:.1f}')),
            ('Lasting 2--4 d (\\%)', lambda s, c: f"{A[s][c]['reference']['pct_2_4d']:.1f}"),
            ('Lasting 5--9 d (\\%)', lambda s, c: f"{A[s][c]['reference']['pct_5_9d']:.1f}"),
            ('Lasting $\\geq 10$ d (\\%)', lambda s, c: ci(s, c, 'pct_ge10d', '{:.2f}')),
            ('Longest (d)', lambda s, c: f"{A[s][c]['reference']['max_length']}"),
            ('Median daily step (km)', lambda s, c: f"{A[s][c]['reference']['median_step_km']:.0f}"),
            ('Mean depth (hPa)', lambda s, c: f"{A[s][c]['per_track']['mean_depth']:.1f}"),
            ('Median footprint ($10^6$ km$^2$)', lambda s, c: f"{A[s][c]['per_track']['median_area']:.2f}"),
            ('\\quad equivalent radius (km)', lambda s, c: f"{A[s][c]['per_track']['req_of_median_area_km']:.0f}"),
            ('Deep snapshots in multi-day trajectories (\\%)', lambda s, c: f"{100 * A[s][c]['share_of_deep_snapshots_in_multiday_tracks']:.1f}")]
    lines = [r'\begin{tabular}{@{}lrrcrr@{}}', r'\toprule',
             r' & \multicolumn{2}{c}{1-cyclones} & & \multicolumn{2}{c}{1-anticyclones} \\', r'\cmidrule{2-3}\cmidrule{5-6}',
             r' & R1, 1979--2023 & ERA5, 1979--2022 & & R1, 1979--2023 & ERA5, 1979--2022 \\', r'\midrule']
    for lab, fn in rows:
        lines.append(' & '.join([lab, fn('R1', 'C'), fn('ERA5', 'C'), '', fn('R1', 'A'), fn('ERA5', 'A')]) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    write('tab_h_tracks.tex', '\n'.join(lines) + '\n')
    for s in A:
        for c in 'CA':
            NUM[f'tracks_{s}_{c}'] = dict(reference={k: v for k, v in A[s][c]['reference'].items() if k != 'survival'},
                                          ci95=A[s][c]['ci95'], per_track=A[s][c]['per_track'], nn=A[s][c]['nn'],
                                          longest=A[s][c]['longest'],
                                          deep_in_multiday=A[s][c]['share_of_deep_snapshots_in_multiday_tracks'])
        NUM[f'P_high_footprint_larger_{s}'] = A[s]['P_high_footprint_larger']


# ------------------------------------------------------------------ reanalysis against the nulls
TRACK_ROWS = [('mean_length', r'Mean trajectory length (d)', '{:.2f}'),
              ('pct_1d', r'Single-day trajectories (\%)', '{:.1f}'),
              ('pct_ge5d', r'Trajectories $\geq 5$ d (\%)', '{:.1f}'),
              ('pct_ge10d', r'Trajectories $\geq 10$ d (\%)', '{:.2f}'),
              ('median_step_km', r'Median daily step (km)', '{:.0f}')]


def track_stat(src, c, k, lab='reference'):
    r = A[src][c][lab][k]
    d = {'real': r}
    for kind in ('ft', 'iaaft'):
        e = A[src]['surrogates'][kind][c][lab][k]
        d[kind] = dict(mean=e['mean'], min=e['min'], max=e['max'], n=e['n'], ratio=r / e['mean'],
                       outside=bool(r > e['max'] or r < e['min']))
    return d


def track_established(c, k, tol=0.10):
    ref = [track_stat(src, c, k)[kind] for src in A for kind in ('ft', 'iaaft')]
    labs = [lab for lab in A['R1'][c] if lab in A['R1']['surrogates']['ft'][c]]
    allr = np.array([track_stat(src, c, k, lab)[kind]['ratio'] for src in A for lab in labs for kind in ('ft', 'iaaft')])
    NUM[f'trkrange_{k}_{c}'] = [float(allr.min()), float(allr.max()), len(labs)]
    return (all(s['outside'] for s in ref) and len(set(np.sign(allr - 1))) == 1 and bool((np.abs(allr - 1) > tol).all()))


def null_table(c):
    lines = [r'\begin{tabular}{@{}lrrrrrcrrrc@{}}', r'\toprule',
             r' & \multicolumn{5}{c}{NCEP--NCAR R1, 1979--2023} & & \multicolumn{3}{c}{ERA5, 1979--2022} & \\',
             r'\cmidrule{2-6}\cmidrule{8-10}',
             r'Property & Rean. & G & AA & R/G & R/AA & & Rean. & R/G & R/AA & Est. \\', r'\midrule']

    def line(lab, fmt, a, b, est):
        return ' & '.join([lab, fmt.format(a['real']), fmt.format(a['ft']['mean']), fmt.format(a['iaaft']['mean']),
                           f"{a['ft']['ratio']:.2f}", f"{a['iaaft']['ratio']:.2f}", '', fmt.format(b['real']),
                           f"{b['ft']['ratio']:.2f}", f"{b['iaaft']['ratio']:.2f}", r'$\bullet$' if est else '']) + r' \\'
    for row in T.ROWS:
        if row is None:
            lines.append(r'\addlinespace')
            continue
        key, lab, fmt = row
        key = key.format(c=c)
        if key.startswith(('trk_', 'mtrk_')):
            continue
        est = T.established(key)
        a, b = T.stat('R1', key), T.stat('ERA5', key)
        NUM[key] = dict(R1=a, ERA5=b, established=est)
        lines.append(line(lab, fmt, a, b, est))
    for k, lab, fmt in TRACK_ROWS:
        est = track_established(c, k)
        a, b = track_stat('R1', c, k), track_stat('ERA5', c, k)
        NUM[f'trk_{k}_{c}'] = dict(R1=a, ERA5=b, established=est)
        lines.append(line(lab, fmt, a, b, est))
    lines += [r'\bottomrule', r'\end{tabular}']
    write(f'tab_h_null_{c}.tex', '\n'.join(lines) + '\n')


def tracksens_table():
    order = ['R = 500 km', 'R = 750 km', 'reference', 'R = 1250 km', 'R = 1500 km', None, 'lam = 0.005', 'lam = 0.02', None,
             'thresholds 10/10', 'thresholds 3/10', 'thresholds 7.5/15', 'thresholds 10/20']
    label = {'R = 500 km': r'$R=500$ km', 'R = 750 km': r'$R=750$ km', 'reference': r'reference',
             'R = 1250 km': r'$R=1250$ km', 'R = 1500 km': r'$R=1500$ km', 'lam = 0.005': r'$\lambda=0.005$ hPa\,km$^{-1}$',
             'lam = 0.02': r'$\lambda=0.02$ hPa\,km$^{-1}$', 'thresholds 10/10': r'$\tau_{\mathrm{lo}}/\tau_{\mathrm{hi}}=10/10$ hPa',
             'thresholds 3/10': r'$3/10$ hPa', 'thresholds 7.5/15': r'$7.5/15$ hPa', 'thresholds 10/20': r'$10/20$ hPa'}
    lines = [r'\begin{tabular}{@{}lrrrrrcrrrrr@{}}', r'\toprule',
             r' & \multicolumn{5}{c}{1-cyclones} & & \multicolumn{5}{c}{1-anticyclones} \\', r'\cmidrule{2-6}\cmidrule{8-12}',
             r' & \multicolumn{2}{c}{Reanalysis} & \multicolumn{3}{c}{R/AA} & & \multicolumn{2}{c}{Reanalysis} & \multicolumn{3}{c}{R/AA} \\',
             r'Setting & 1 d (\%) & length (d) & $\geq 5$ d & $\geq 10$ d & step & & 1 d (\%) & length (d) & $\geq 5$ d & $\geq 10$ d & step \\',
             r'\midrule']
    for lab in order:
        if lab is None:
            lines.append(r'\addlinespace')
            continue
        cells = [label[lab]]
        for c in 'CA':
            r = A['R1'][c][lab]
            cells += [f"{r['pct_1d']:.1f}", f"{r['mean_length']:.2f}"]
            cells += [f"{track_stat('R1', c, k, lab)['iaaft']['ratio']:.2f}" for k in ('pct_ge5d', 'pct_ge10d', 'median_step_km')]
            if c == 'C':
                cells.append('')
        lines.append(' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    write('tab_h_tracksens.tex', '\n'.join(lines) + '\n')


# ------------------------------------------------------------------ detection criteria
def detectors_table():
    D = {c: json.load(open(os.path.join(ROOT, 'output', 'experiments', 'beyond_amplitude', f'detectors_monthly_{t}.json')))
         for c, t in (('C', 'sup'), ('A', 'sub'))}
    names = [('P', 'Topological depth (prominence)'), ('A', 'Amplitude of the extremum'),
             ('AN', r'Amplitude, no stronger extremum within 1000 km'),
             ('L', r'Depth relative to a 500--1000 km annulus'), ('C', r'Closed-contour depth, contour within 1200 km')]
    lines = [r'\begin{tabular}{@{}lrrrcrrr@{}}', r'\toprule',
             r' & \multicolumn{3}{c}{1-cyclones} & & \multicolumn{3}{c}{1-anticyclones} \\', r'\cmidrule{2-4}\cmidrule{6-8}',
             r'Criterion & In $\geq 3$ d (\%) & Difference [95\,\% CI] & Terrain (\%) & & In $\geq 3$ d (\%) & Difference [95\,\% CI] & Terrain (\%) \\',
             r'\midrule']
    for k, lab in names:
        cells = [lab]
        for c in 'CA':
            d = D[c][k]
            v, lo, hi = d['single']['ge3_minus_P']
            NUM[f'detector_{k}_{c}'] = dict(ge3=d['single']['ge3'], diff=[v, lo, hi], two_level_diff=d['two_level']['ge3_minus_P'],
                                            terrain=d['pct_hgt_gt1500'], pct1_single=d['single']['pct1'],
                                            pct1_two=d['two_level']['pct1'])
            cells += [f"{d['single']['ge3']:.1f}", '--' if k == 'P' else f'${v:+.1f}$ [${lo:+.1f}$, ${hi:+.1f}$]',
                      f"{d['pct_hgt_gt1500']:.1f}"]
            if c == 'C':
                cells.append('')
        lines.append(' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    write('tab_h_detectors.tex', '\n'.join(lines) + '\n')


def main():
    renamed_copies()
    tracks_table()
    for c in 'CA':
        null_table(c)
    tracksens_table()
    detectors_table()
    for s in A:
        for c in 'CA':
            NUM[f'climatology_{s}_{c}'] = A[s][c]['climatology']
            NUM[f'monthly_{s}_{c}'] = A[s][c]['monthly']
    NUM['event_2012'] = A['R1']['event_2012']
    json.dump(NUM, open(os.path.join(TAB, 'numbers_hybrid.json'), 'w'), indent=1, default=float)
    print('established:', sorted(k for k, v in NUM.items() if isinstance(v, dict) and v.get('established')))
    print('not established:', sorted(k for k, v in NUM.items() if isinstance(v, dict) and v.get('established') is False))
    print({k: [round(x, 2) for x in v[:2]] for k, v in NUM.items() if k.startswith('trkrange')})


if __name__ == '__main__':
    main()
