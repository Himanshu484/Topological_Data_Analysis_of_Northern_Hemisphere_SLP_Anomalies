"""
Terrain check of the daily analysis: the decision rule of the paper applied
with and without features over terrain higher than 1500 m.

    cd scripts/experiments/gaussian_null && python terrain_check.py && GN_SOURCE=era5 python terrain_check.py && cd ../../..
    python scripts/paper/make_terrain_outputs.py

Writes <manuscript>/tables/tab_h_terrain.tex and numbers_terrain.json (see paths.py).

The rule (Section 2.7): in both reanalyses and against both nulls the
reanalysis lies outside every realisation, the sign is the same, and the
difference from the ensemble mean exceeds 5 % (10 % for day-to-day
statistics); for trajectory statistics this must hold for every tracker
setting.  'full' is the field as analysed in the paper, 'low' the same with
every feature over terrain above 1500 m removed from reanalysis and nulls.
"""
import json
import os

import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402

ROOT, TAB = paths.ROOT, paths.TABLES
D = {'R1': json.load(open(os.path.join(ROOT, 'output', 'experiments', 'gaussian_null', 'terrain_check.json'))),
     'ERA5': json.load(open(os.path.join(ROOT, 'output', 'experiments', 'gaussian_null_era5', 'terrain_check.json')))}
META = D['R1']['_meta']
LAGS = META['lags']
CONFIGS = META['configs']
TOL = {'field': 0.05, 'step': 0.10}
NAME = {'C': '1-cyclones', 'A': '1-anticyclones'}


def members(src, kind):
    return [n for n in sorted(D[src]) if n.startswith(kind + '_')]


def compare(src, kind, v, c, fn):
    """Reanalysis value, surrogate values (those for which fn is defined), outside flag, relative difference."""
    r = fn(D[src]['real'][v][c])
    e = np.array([x for x in (fn(D[src][n][v][c]) for n in members(src, kind)) if x is not None], float)
    return r, e, bool(r < e.min() or r > e.max()), float(r / e.mean() - 1) if e.mean() != 0 else np.inf


def verdict(v, c, fns, cls):
    """fns: one getter, or one per tracker setting (criterion iv)."""
    out, rel, ratio_aa = [], [], {}
    for src in ('R1', 'ERA5'):
        ra = []
        for fn in fns:
            for kind in ('ft', 'iaaft'):
                r, e, o, d = compare(src, kind, v, c, fn)
                out.append(o); rel.append(d)
                if kind == 'iaaft':
                    ra.append(1 + d)
        ratio_aa[src] = [float(min(ra)), float(max(ra))]
    rel = np.array(rel)
    return dict(established=bool(all(out) and (np.all(rel > TOL[cls]) or np.all(rel < -TOL[cls]))), outside_all=bool(all(out)),
                same_sign=bool(np.all(rel > 0) or np.all(rel < 0)), min_abs_rel=float(np.abs(rel).min()),
                max_abs_rel=float(np.abs(rel).max()), ratio_aa=ratio_aa)


def g(key):
    return lambda s: s[key]


def coh(lag):
    i = LAGS.index(lag)
    return lambda s: s['coh500'][i] - s['coh500'][-1]             # in excess of chance (lag of 365 days)


def trk(key):
    return [(lambda s, lab=lab: s['tracks'][lab][key] if lab in s['tracks'] else None) for lab in CONFIGS]


ROWS = [('N5', r'$N_{5}$', [g('N5')], 'field'), ('N10', r'$N_{10}$', [g('N10')], 'field'),
        ('N20', r'$N_{20}$', [g('N20')], 'field'), ('N30', r'$N_{30}$', [g('N30')], 'field'),
        ('TP2s', r'$\mathrm{TP}_2$', [g('TP2s')], 'field'), None,
        ('half_med_10', r'Footprint area, median', [g('half_med_10')], 'field'),
        ('mhalf_med', r'\quad at matched count', [g('mhalf_med')], 'field'),
        ('slope_half', r'Area--depth exponent', [g('slope_half')], 'field'),
        ('nest_frac_10', r'Share nested', [g('nest_frac_10')], 'field'),
        ('nest_ge2_10', r'Share at nesting level $\geq 2$', [g('nest_ge2_10')], 'field'), None,
        ('coh1', r'Coherence in excess of chance, 1 d', [coh(1)], 'step'),
        ('coh2', r'\quad 2 d', [coh(2)], 'step'), ('coh3', r'\quad 3 d', [coh(3)], 'step'), ('coh5', r'\quad 5 d', [coh(5)], 'step'), None,
        ('mean_length', r'Mean trajectory length$^{*}$', trk('mean_length'), 'step'),
        ('pct_ge5d', r'Trajectories $\geq 5$ d$^{*}$', trk('pct_ge5d'), 'step'),
        ('pct_ge10d', r'Trajectories $\geq 10$ d$^{*}$', trk('pct_ge10d'), 'step'),
        ('median_step_km', r'Median daily step$^{*}$', trk('median_step_km'), 'step'),
        ('pct_1d', r'Single-day trajectories$^{*}$', trk('pct_1d'), 'step')]


def cell(rng):
    a, b = rng
    return f'{a:.2f}' if round(a, 2) == round(b, 2) else (f'{a:.2f}--{b:.2f}' if b < 9.995 else f'{a:.1f}--{b:.0f}')


def main():
    NUM = {'limit_m': META['limit_m'], 'verdict': {}, 'changed': []}
    lines = [r'\begin{tabular}{@{}lrrcrrcc@{\hspace{0.6em}}c@{}}', r'\toprule',
             r' & \multicolumn{2}{c}{R1, R/AA} & & \multicolumn{2}{c}{ERA5, R/AA} & & \multicolumn{2}{c}{Established} \\',
             r'\cmidrule{2-3}\cmidrule{5-6}\cmidrule{8-9}', r'Property & all & masked & & all & masked & & all & masked \\']
    for c in 'CA':
        NUM['verdict'][c] = {}
        lines += [r'\midrule', r'\multicolumn{9}{@{}l}{\emph{' + NAME[c] + r'}} \\']
        for row in ROWS:
            if row is None:
                lines.append(r'\addlinespace')
                continue
            key, lab, fns, cls = row
            V = {v: verdict(v, c, fns, cls) for v in ('full', 'low')}
            NUM['verdict'][c][key] = V
            if V['full']['established'] != V['low']['established']:
                NUM['changed'].append(f'{c}:{key}')
            mk = {v: r'$\bullet$' if V[v]['established'] else '' for v in V}
            lines.append(' & '.join([lab, cell(V['full']['ratio_aa']['R1']), cell(V['low']['ratio_aa']['R1']), '',
                                     cell(V['full']['ratio_aa']['ERA5']), cell(V['low']['ratio_aa']['ERA5']), '', mk['full'], mk['low']]) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    open(os.path.join(TAB, 'tab_h_terrain.tex'), 'w').write('\n'.join(lines) + '\n')
    for src in D:                                               # how much of the field the mask removes
        NUM[src] = {c: dict(share_high_10=D[src]['real']['full'][c]['share_high_10'],
                            share_high_10_aa=float(np.mean([D[src][n]['full'][c]['share_high_10'] for n in members(src, 'iaaft')])),
                            N10_full=D[src]['real']['full'][c]['N10'], N10_low=D[src]['real']['low'][c]['N10'],
                            n_surrogates={k: len(members(src, k)) for k in ('ft', 'iaaft')}) for c in 'CA'}
    json.dump(NUM, open(os.path.join(TAB, 'numbers_terrain.json'), 'w'), indent=1, default=float)
    print('wrote tab_h_terrain.tex; verdicts that change with the mask:', NUM['changed'] or 'none')
    for c in 'CA':
        for key, V in NUM['verdict'][c].items():
            print(f"{c} {key:15s} full est={V['full']['established']!s:5} |rel| {V['full']['min_abs_rel']:.3f}-{V['full']['max_abs_rel']:.3f}"
                  f"   low est={V['low']['established']!s:5} |rel| {V['low']['min_abs_rel']:.3f}-{V['low']['max_abs_rel']:.3f}")


if __name__ == '__main__':
    main()
