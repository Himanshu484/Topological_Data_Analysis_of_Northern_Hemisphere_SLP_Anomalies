"""
Tables and figure of the six-hourly section of the manuscript.

    python scripts/experiments/era5_6h/analyse6h.py        # metrics_6h.json
    python scripts/paper/make_tables.py                    # numbers_hybrid.json (daily ratios, for the comparison columns)
    python scripts/paper/make_6h_outputs.py

Writes <manuscript>/tables/tab_h_6h_{null_C,null_A,tracks,coherence}.tex and numbers_6h.json,
and <manuscript>/figures/fig_h_6h.pdf (see paths.py).

Data sets in metrics_6h.json, named <name>_<res>:
  res 1p5   ERA5 at 1.5 degrees: real, ft_r*, iaaft_r*, ftw_r0 (control without seams)
  res r1    NCEP-NCAR R1 at 2.5 degrees: the same
  res 0p25  ERA5 at 0.25 degrees: real only
Ratios to the nulls are formed within one res.  A difference is marked as
established by the rule of the paper (Section 2.7) applied to the two
reanalyses: outside every realisation of both nulls in both, same sign, and
beyond the tolerance (5 %; 10 % for step-to-step statistics); for trajectory
statistics, at every linking radius.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402

ROOT, TAB, FIG = paths.ROOT, paths.TABLES, paths.FIGURES
DATA = os.path.join(ROOT, 'data', 'processed_data', 'experiments', 'era5_6h')
M = json.load(open(os.path.join(ROOT, 'output', 'experiments', 'era5_6h', 'metrics_6h.json')))
META = M.pop('_meta')
for _n in M:                                                   # tracker-free loss rate: no successor within 500 km after one step
    for _c in 'CA':
        M[_n][_c]['lost6h'] = 100 * (1 - M[_n][_c]['coherence']['500'][0])
        for _v in ('matched', 'low'):
            if _v in M[_n][_c]:
                M[_n][_c][_v]['lost6h'] = 100 * (1 - M[_n][_c][_v]['coherence']['500'][0])
DAILY = json.load(open(os.path.join(TAB, 'numbers_hybrid.json')))
NAME = {'C': '1-cyclones', 'A': '1-anticyclones'}
COL = {'real': 'k', 'ft': '#1f77b4', 'iaaft': '#d95f02'}
# (label in DAILY and in the tables, res key, LaTeX label, line style in the figure)
SOURCES = [s for s in [('ERA5', '1p5', r'ERA5, $1.5^\circ$', '-'), ('R1', 'r1', r'R1, $2.5^\circ$', '-.')] if f'real_{s[1]}' in M]
HI = 'real_0p25' if 'real_0p25' in M else None
RADII = [str(r) for r in META['track_R_km']]
REF_R = '500'
TOL = {'field': 0.05, 'step': 0.10}
NUM = {'sources': [s[0] for s in SOURCES], 'has_0p25': HI is not None}

plt.rcParams.update({'font.size': 8.5, 'axes.titlesize': 9, 'axes.labelsize': 8.5, 'legend.fontsize': 7,
                     'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'pdf.fonttype': 42, 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03})


def get(d, path):
    for p in path:
        d = d[p]
    return d


def members(kind, res):
    return [n for n in sorted(M) if n.startswith(kind + '_r') and n.endswith('_' + res)]


def ens(kind, c, path, res, need=None):
    return np.array([get(M[n][c], path) for n in members(kind, res) if need is None or need in M[n][c]], float)


def mm(e):
    return dict(mean=float(e.mean()), min=float(e.min()), max=float(e.max()), n=len(e)) if len(e) else None


def rr(r, e):
    return float(r / e.mean()) if len(e) and e.mean() > 0 else None


def stat(c, path, res):
    r = get(M[f'real_{res}'][c], path)
    d = {'real': r}
    for kind in ('ft', 'iaaft'):
        e = ens(kind, c, path, res)
        d[kind] = dict(mm(e), ratio=rr(r, e), outside=bool(r < e.min() or r > e.max())) if len(e) else None
    return d


def verdict(c, key, cls, radii=None, pre=()):
    """The rule of the paper over both reanalyses and both nulls; radii: trajectory statistic tested at each;
    pre: prefix of the path, ('low',) for the terrain-masked variant."""
    if len(SOURCES) < 2:
        return None
    rel, outside = [], []
    for _, res, _, _ in SOURCES:
        for path in ([pre + (key,)] if radii is None else [pre + ('tracks', R, key) for R in radii]):
            r = get(M[f'real_{res}'][c], path)
            for kind in ('ft', 'iaaft'):
                e = ens(kind, c, path, res)
                outside.append(bool(r < e.min() or r > e.max()))
                rel.append(r / e.mean() - 1 if e.mean() > 0 else (np.inf if r > 0 else 0.0))
    rel = np.array(rel)
    return dict(established=bool(all(outside) and (np.all(rel > TOL[cls]) or np.all(rel < -TOL[cls]))),
                outside_all=bool(all(outside)), same_sign=bool(np.all(rel > 0) or np.all(rel < 0)),
                min_abs_rel=float(np.min(np.abs(rel))), max_abs_rel=float(np.max(np.abs(rel[np.isfinite(rel)]))),
                sign=int(np.sign(np.median(rel))))


def track_verdict(c, key, pre=()):
    """Verdict at every radius, and for radii from 250 and from 375 km upward."""
    v = verdict(c, key, 'step', RADII, pre)
    if v is not None:
        for rmin in (250, 375):
            w = verdict(c, key, 'step', [R for R in RADII if int(R) >= rmin], pre)
            v[f'established_R{rmin}'] = w['established']
            v[f'min_abs_rel_R{rmin}'], v[f'max_abs_rel_R{rmin}'] = w['min_abs_rel'], w['max_abs_rel']
    return v


def mark(v):
    if v is None:
        return ''
    return r'$\bullet$' if v.get('established') else (r'$\circ$' if v.get('established_R375') else '')


def write(name, lines):
    open(os.path.join(TAB, name), 'w').write('\n'.join(lines) + '\n')
    print('wrote', name)


def fd(fmt, v):
    return '--' if v is None else fmt.format(v)


ROWS = [('Nall', 'All extrema', '{:.0f}', None, 'field'),
        ('N5_ANN', r'$N_{5}$', '{:.2f}', 'N5_{c}_ANN', 'field'),
        ('N10_ANN', r'$N_{10}$', '{:.2f}', 'N10_{c}_ANN', 'field'),
        ('N20_ANN', r'$N_{20}$', '{:.2f}', 'N20_{c}_ANN', 'field'),
        ('N30_ANN', r'$N_{30}$', '{:.3f}', 'N30_{c}_ANN', 'field'),
        ('TP2', r'$\mathrm{TP}_2$ (hPa$^2$)', '{:.0f}', 'TP2_{c}', 'field'),
        None,
        ('half_med_10', r'Footprint ($10^6$ km$^2$)', '{:.2f}', 'half_med_{c}_10', 'field'),
        ('mhalf_med', r'\quad matched count', '{:.2f}', 'mhalf_med_{c}', 'field'),
        ('nest_frac_10', r'Nested', '{:.3f}', 'nest_frac_{c}_10', 'field'),
        ('nest_ge2_10', r'Nested, level $\geq 2$', '{:.3f}', 'nest_ge2_{c}_10', 'field'),
        None,
        ('lost6h', r'Lost within 6 h (\%)', '{:.1f}', None, 'step')]
LOW_ROWS = [('N10', r'$N_{10}$', '{:.2f}', 'field'), ('lost6h', r'Lost within 6 h (\%)', '{:.1f}', 'step')]
LOW_TRACK_ROWS = [('mean_days', r'Mean duration (d)', '{:.2f}'), ('pct_ge5d', r'Traj.\ $\geq 5$ d (\%)', '{:.2f}')]
TRACK_ROWS = [('pct_single_step', r'One-field traj.\ (\%)', '{:.1f}'),
              ('mean_days', r'Mean duration (d)', '{:.2f}'),
              ('pct_ge5d', r'Traj.\ $\geq 5$ d (\%)', '{:.2f}'),
              ('median_step_km', r'Median step (km)', '{:.0f}')]


def null_table():
    ns = len(SOURCES)
    for c in 'CA':
        NUM.setdefault('verdict', {})[c] = {}
        head1 = r' & $0.25^\circ$'
        head2 = 'Property & Rean.'
        rules = []
        for i, (_, _, lab, _) in enumerate(SOURCES):
            head1 += r' & & \multicolumn{6}{c}{' + lab + ', six-hourly}'
            head2 += r' & & Rean. & G & AA & R/G & R/AA & daily'
            rules.append(r'\cmidrule{%d-%d}' % (4 + 7 * i, 9 + 7 * i))
        lines = [r'\begin{tabular}{@{}lr' + 'crrrrrr' * ns + r'c@{}}', r'\toprule', head1 + r' & \\', ''.join(rules), head2 + r' & \\', r'\midrule']

        def row(lab, fmt, path, dkey, v):
            cells = [lab, fd(fmt, get(M[HI][c], path) if HI and path[0] != 'low' else None)]
            for src, res, _, _ in SOURCES:
                s = stat(c, path, res)
                NUM.setdefault(src, {})['_'.join(path) + '_' + c] = s
                daily = '--'
                if dkey and dkey.format(c=c) in DAILY and src in DAILY[dkey.format(c=c)]:
                    daily = f"{DAILY[dkey.format(c=c)][src]['iaaft']['ratio']:.2f}"
                g, a = s['ft'], s['iaaft']
                cells += ['', fmt.format(s['real']), fd(fmt, g and g['mean']), fd(fmt, a and a['mean']),
                          fd('{:.2f}', g and g['ratio']), fd('{:.2f}', a and a['ratio']), daily]
            lines.append(' & '.join(cells + [mark(v)]) + r' \\')

        for r in ROWS:
            if r is None:
                lines.append(r'\addlinespace')
                continue
            key, lab, fmt, dkey, cls = r
            v = verdict(c, key, cls) if key != 'Nall' else None       # grid-scale extrema: not assessed, as in the daily analysis
            NUM['verdict'][c][key] = v
            row(lab, fmt, (key,), dkey, v)
        for key, lab, fmt in TRACK_ROWS:
            v = track_verdict(c, key)
            NUM['verdict'][c][key] = v
            row(lab, fmt, ('tracks', REF_R, key), None, v)
        NUM['verdict'][c]['mean_step_km'] = track_verdict(c, 'mean_step_km')
        if all('low' in M[n][c] for n in M if not n.endswith('_0p25')):
            lines += [r'\addlinespace', r'\multicolumn{%d}{@{}l}{\emph{Excluding terrain above $1500$ m}} \\' % (3 + 7 * ns)]
            for key, lab, fmt, cls in LOW_ROWS:
                v = verdict(c, key, cls, pre=('low',))
                NUM['verdict'][c]['low_' + key] = v
                row(r'\quad ' + lab, fmt, ('low', key), None, v)
            for key, lab, fmt in LOW_TRACK_ROWS + [('mean_step_km', '', ''), ('median_step_km', '', ''), ('pct_single_step', '', '')]:
                v = track_verdict(c, key, pre=('low',))
                NUM['verdict'][c]['low_' + key] = v
                if lab:
                    row(r'\quad ' + lab, fmt, ('low', 'tracks', REF_R, key), None, v)
            for src, res, _, _ in SOURCES:                        # terrain-masked trajectory ratios at every radius, for the text
                NUM[src][f'low_tracks_{c}'] = {R: {key: dict(real=M[f'real_{res}'][c]['low']['tracks'][R][key],
                                                             ratio_aa=rr(M[f'real_{res}'][c]['low']['tracks'][R][key], ens('iaaft', c, ('low', 'tracks', R, key), res)),
                                                             ratio_g=rr(M[f'real_{res}'][c]['low']['tracks'][R][key], ens('ft', c, ('low', 'tracks', R, key), res)))
                                                       for key in ('mean_days', 'pct_ge5d', 'mean_step_km', 'median_step_km', 'pct_single_step')} for R in RADII}
                ex = np.array(M[f'real_{res}'][c]['low']['coherence']['500'])
                NUM[src][f'low_coherence_ratio_{c}'] = {kind: [float(x) for x in (ex[:-1] - ex[-1]) / np.mean([np.array(M[n][c]['low']['coherence']['500'])[:-1] - M[n][c]['low']['coherence']['500'][-1] for n in members(kind, res)], 0)] for kind in ('ft', 'iaaft')}
        lines += [r'\bottomrule', r'\end{tabular}']
        write(f'tab_h_6h_null_{c}.tex', lines)


def tracks_table():
    """Each reanalysis against its AA null and the count-matched variant, for every linking radius."""
    cols = [('pct_single_step', '{:.1f}'), ('mean_days', '{:.2f}'), ('pct_ge5d', '{:.2f}'), ('median_step_km', '{:.0f}')]
    extra = ['mean_step_km', 'pct_ge10d', 'per_year', 'pct_lt1d']
    lines = [r'\begin{tabular}{@{}lrrrcrrrcrrrcrrr@{}}', r'\toprule',
             r' & \multicolumn{3}{c}{1 field (\%)} & & \multicolumn{3}{c}{Mean duration (d)} & & \multicolumn{3}{c}{$\geq 5$ d (\%)} & & \multicolumn{3}{c}{Median step (km)} \\',
             r'\cmidrule{2-4}\cmidrule{6-8}\cmidrule{10-12}\cmidrule{14-16}',
             r'$R$ (km) & Rean. & AA & AA$_m$ & & Rean. & AA & AA$_m$ & & Rean. & AA & AA$_m$ & & Rean. & AA & AA$_m$ \\']
    for src, res, lab, _ in SOURCES:
        for c in 'CA':
            lines += [r'\midrule', r'\multicolumn{16}{@{}l}{\emph{' + lab + ', ' + NAME[c] + r'}} \\']
            for R in RADII:
                cells, rec = [R], {}
                for key in [k for k, _ in cols] + extra:
                    r = M[f'real_{res}'][c]['tracks'][R][key]
                    aa, g = ens('iaaft', c, ('tracks', R, key), res), ens('ft', c, ('tracks', R, key), res)
                    am = ens('iaaft', c, ('matched', 'tracks', R, key), res, need='matched')
                    gm = ens('ft', c, ('matched', 'tracks', R, key), res, need='matched')
                    rec[key] = dict(real=r, aa=mm(aa), aa_matched=mm(am), g=mm(g), g_matched=mm(gm), ratio_aa=rr(r, aa),
                                    ratio_aa_matched=rr(r, am), ratio_g=rr(r, g), ratio_g_matched=rr(r, gm))
                    if res == '1p5' and HI:
                        rec[key]['hi'] = M[HI][c]['tracks'][R][key]
                    if f'ftw_r0_{res}' in M:
                        w = M[f'ftw_r0_{res}'][c]['tracks'][R][key]
                        rec[key]['ftw'], rec[key]['ratio_ftw'] = w, (r / w if w > 0 else None)
                for key, fmt in cols:
                    cells += [fmt.format(rec[key]['real']), fd(fmt, rec[key]['aa'] and rec[key]['aa']['mean']),
                              fd(fmt, rec[key]['aa_matched'] and rec[key]['aa_matched']['mean']), '']
                NUM.setdefault(src, {})[f'tracks_R{R}_{c}'] = rec
                lines.append(' & '.join(cells[:-1]) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    write('tab_h_6h_tracks.tex', lines)


def excess(c, name, radius='500', matched=False):
    v = np.array((M[name][c]['matched'] if matched else M[name][c])['coherence'][radius])
    return v[:-1] - v[-1], v


def coherence_table(radius='500'):
    lags = META['lags_steps'][:-1]
    lab = {1: '6 h', 2: '12 h', 4: '1 d', 8: '2 d', 12: '3 d', 20: '5 d', 28: '7 d'}
    lines = [r'\begin{tabular}{@{}lrrrrrcrrrrr@{}}', r'\toprule',
             r' & \multicolumn{5}{c}{1-cyclones} & & \multicolumn{5}{c}{1-anticyclones} \\', r'\cmidrule{2-6}\cmidrule{8-12}',
             r' & \multicolumn{2}{c}{$c_k$} & \multicolumn{3}{c}{Excess ratio} & & \multicolumn{2}{c}{$c_k$} & \multicolumn{3}{c}{Excess ratio} \\',
             r'Lag & Rean. & AA & R/G & R/AA & R/AA$_m$ & & Rean. & AA & R/G & R/AA & R/AA$_m$ \\']
    for src, res, slab, _ in SOURCES:
        rec = {}
        lines += [r'\midrule', r'\multicolumn{12}{@{}l}{\emph{' + slab + r'}} \\']
        for i, k in enumerate(lags + [META['lags_steps'][-1]]):
            cells = [lab.get(k, r'1 yr (chance)')]
            for c in 'CA':
                ex_r, v_r = excess(c, f'real_{res}', radius)
                sur = {kind: [excess(c, n, radius) for n in members(kind, res)] for kind in ('ft', 'iaaft')}
                sur['iaaft_m'] = [excess(c, n, radius, True) for n in members('iaaft', res) if 'matched' in M[n][c]]
                aa_v = np.mean([s[1] for s in sur['iaaft']], 0) if sur['iaaft'] else None
                ratios = {kind: (float(ex_r[i] / np.mean([s[0] for s in sur[kind]], 0)[i]) if sur[kind] and i < len(lags) else None) for kind in sur}
                if f'ftw_r0_{res}' in M and i < len(lags):
                    ratios['ftw'] = float(ex_r[i] / excess(c, f'ftw_r0_{res}', radius)[0][i])
                rec[f'{c}_{k}'] = dict(real=float(v_r[i]), aa=None if aa_v is None else float(aa_v[i]), ratio=ratios,
                                       hi=float(M[HI][c]['coherence'][radius][i]) if HI and res == '1p5' else None)
                cells += [f'{v_r[i]:.3f}', fd('{:.3f}', None if aa_v is None else float(aa_v[i])),
                          fd('{:.2f}', ratios['ft']), fd('{:.2f}', ratios['iaaft']), fd('{:.2f}', ratios['iaaft_m'])]
                if c == 'C':
                    cells.append('')
            if k == META['lags_steps'][-1]:
                lines.append(r'\addlinespace')
            lines.append(' & '.join(cells) + r' \\')
        NUM.setdefault(src, {})[f'coherence_{radius}km'] = rec
    lines += [r'\bottomrule', r'\end{tabular}']
    write('tab_h_6h_coherence.tex', lines)


def figure():
    taus = META['taus']
    kname = {'ft': 'Gaussian', 'iaaft': 'amplitude-adjusted'}
    fig, axs = plt.subplots(3, 2, figsize=(7.0, 7.8))
    days = np.arange(1, 31)
    lag_d = np.array(META['lags_steps'][:-1]) / 4.0
    for j, c in enumerate('CA'):
        # ---- (a, b) counts: ratio to each null
        ax = axs[0, j]
        for kind in ('ft', 'iaaft'):
            for src, res, _, ls in SOURCES:
                E = np.array([[M[n][c][f'N{t:g}_ANN'] for t in taus] for n in members(kind, res)])
                r = np.array([M[f'real_{res}'][c][f'N{t:g}_ANN'] for t in taus])
                ax.fill_between(taus, r / E.max(0), r / E.min(0), color=COL[kind], alpha=0.2, lw=0)
                ax.plot(taus, r / E.mean(0), ls, color=COL[kind], lw=1.3,
                        label=f"six-hourly {src} / {kname[kind]} null")
            d = [DAILY.get(f'N{t:g}_{c}_ANN') for t in (2, 5, 10, 20, 30)]
            if all(d):
                ax.plot([2, 5, 10, 20, 30], [x['ERA5'][kind]['ratio'] for x in d], 'o:', color=COL[kind], ms=2.6, lw=0.8,
                        label=f"daily ERA5, 2.5° / {kname[kind]} null")
        ax.axhline(1, color='k', lw=0.6)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xticks(taus); ax.set_xticklabels([f'{t:g}' for t in taus])
        yt = [0.3, 0.5, 0.7, 1, 1.5, 2, 3]
        ax.set_yticks(yt); ax.set_yticklabels([f'{y:g}' for y in yt]); ax.minorticks_off(); ax.set_ylim(0.2, 3.2)
        ax.set_xlabel(r'depth threshold $\tau$ (hPa)'); ax.set_title(NAME[c])
        if j == 0:
            ax.set_ylabel('number of features: ratio to the null')
            ax.legend(frameon=False, fontsize=5.6, loc='lower left')
        ax.text(-0.07, 1.04, 'ab'[j], transform=ax.transAxes, fontweight='bold', fontsize=9.5, ha='right')
        # ---- (c, d) coherence in excess of chance
        ax = axs[1, j]
        for src, res, _, ls in SOURCES:
            ax.plot(lag_d, excess(c, f'real_{res}')[0], ls, marker='o', color='k', ms=2.6, lw=1.3, label=f'{src} reanalysis')
            for kind in ('ft', 'iaaft'):
                S = np.array([excess(c, n)[0] for n in members(kind, res)])
                ax.fill_between(lag_d, S.min(0), S.max(0), color=COL[kind], alpha=0.2, lw=0)
                ax.plot(lag_d, S.mean(0), ls, color=COL[kind], lw=1.0, label=f'{src} {kname[kind]} null')
        if HI:
            ax.plot(lag_d, excess(c, HI)[0], ':', marker='s', color='0.5', ms=2.2, lw=0.9, label='ERA5 reanalysis, 0.25°')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xticks([0.25, 0.5, 1, 2, 3, 5, 7]); ax.set_xticklabels(['6 h', '12 h', '1 d', '2 d', '3 d', '5 d', '7 d']); ax.minorticks_off()
        ax.set_xlabel('lag')
        if j == 0:
            ax.set_ylabel('coherence within 500 km in excess of chance')
            ax.legend(frameon=False, fontsize=5.6, loc='lower left')
        ax.text(-0.07, 1.04, 'cd'[j], transform=ax.transAxes, fontweight='bold', fontsize=9.5, ha='right')
        # ---- (e, f) trajectory survival at the reference radius
        ax = axs[2, j]
        for src, res, _, ls in SOURCES:
            ax.plot(days, M[f'real_{res}'][c]['tracks'][REF_R]['survival_days'], ls, marker='o', color='k', ms=2.3, lw=1.3, label=f'{src} reanalysis')
            for kind in ('ft', 'iaaft'):
                S = ens(kind, c, ('tracks', REF_R, 'survival_days'), res)
                ax.fill_between(days, np.clip(S.min(0), 1e-6, None), np.clip(S.max(0), 1e-6, None), color=COL[kind], alpha=0.2, lw=0)
                ax.plot(days, np.clip(S.mean(0), 1e-6, None), ls, color=COL[kind], lw=1.0, label=f'{src} {kname[kind]} null')
        S = ens('iaaft', c, ('matched', 'tracks', REF_R, 'survival_days'), SOURCES[0][1], need='matched')
        if len(S):
            ax.plot(days, np.clip(S.mean(0), 1e-6, None), '--', color='#7a3a00', lw=0.9, label=f'{SOURCES[0][0]} amplitude-adjusted null, matched counts')
        if HI:
            ax.plot(days, M[HI][c]['tracks'][REF_R]['survival_days'], ':', marker='s', color='0.5', ms=2.0, lw=0.9, label='ERA5 reanalysis, 0.25°')
        ax.set_yscale('log'); ax.set_ylim(1e-4, 1.2); ax.set_xlim(0.5, 20.5); ax.set_xticks([1, 5, 10, 15, 20])
        ax.set_xlabel(r'duration $\ell$ (days)')
        if j == 0:
            ax.set_ylabel(r'share of trajectories lasting $\geq \ell$' + f'\n($R$ = {REF_R} km per 6 h)')
            ax.legend(frameon=False, fontsize=5.6, loc='upper right')
        ax.text(-0.07, 1.04, 'ef'[j], transform=ax.transAxes, fontweight='bold', fontsize=9.5, ha='right')
    fig.tight_layout(h_pad=1.0, w_pad=1.0)
    fig.savefig(os.path.join(FIG, 'fig_h_6h.pdf'))
    plt.close(fig)
    print('wrote fig_h_6h.pdf')


def coherence_verdict(radius='500'):
    """The rule applied to the coherence in excess of chance, lag by lag (10 % tolerance)."""
    lags = META['lags_steps'][:-1]
    out = {}
    for c in 'CA':
        out[c] = {}
        for i, k in enumerate(lags):
            rel, outside = [], []
            for _, res, _, _ in SOURCES:
                r = excess(c, f'real_{res}', radius)[0][i]
                for kind in ('ft', 'iaaft'):
                    e = np.array([excess(c, n, radius)[0][i] for n in members(kind, res)])
                    outside.append(bool(r < e.min() or r > e.max()))
                    rel.append(r / e.mean() - 1)
            rel = np.array(rel)
            out[c][str(k)] = dict(established=bool(all(outside) and (np.all(rel > TOL['step']) or np.all(rel < -TOL['step']))),
                                  min_ratio=float(1 + rel.min()), max_ratio=float(1 + rel.max()))
    NUM['verdict_coherence'] = out


def main():
    null_table()
    if len(SOURCES) == 2:
        coherence_verdict()
    tracks_table()
    coherence_table()
    figure()
    keep = ('Nall', 'N2_ANN', 'N5_ANN', 'N10_ANN', 'N20_ANN', 'N30_ANN', 'TP1', 'TP2', 'half_med_10', 'half_req_km_10',
            'nest_frac_10', 'nest_ge2_10', 'depth_over_amp_median_10', 'lost6h')
    if HI:                                                    # resolution bridge: ERA5 at 0.25 against 1.5 degrees
        NUM['bridge'] = {c: {k: dict(hi=M[HI][c][k], lo=M['real_1p5'][c][k], ratio=M[HI][c][k] / M['real_1p5'][c][k]) for k in keep} for c in 'CA'}
    NUM['terrain'] = {c: {n: M[n][c].get('terrain') for n in ('real_0p25', 'real_1p5', 'iaaft_r0_1p5', 'real_r1', 'iaaft_r0_r1') if n in M} for c in 'CA'}
    for src, res, _, _ in SOURCES:
        S = NUM.setdefault(src, {})
        S['n_realisations'] = {k: len(members(k, res)) for k in ('ft', 'iaaft')}
        S['real'] = {c: {k: M[f'real_{res}'][c][k] for k in keep} for c in 'CA'}
        asym = {}
        for key in ('TP1', 'TP2', 'N10_ANN', 'N20_ANN'):
            obs = M[f'real_{res}']['C'][key] / M[f'real_{res}']['A'][key]
            g = ens('ft', 'C', (key,), res).mean() / ens('ft', 'A', (key,), res).mean()
            aa = ens('iaaft', 'C', (key,), res).mean() / ens('iaaft', 'A', (key,), res).mean()
            asym[key] = dict(obs=float(obs), g=float(g), aa=float(aa), share=float((np.log(aa) - np.log(g)) / (np.log(obs) - np.log(g))))
        S['asymmetry'] = asym
        S['matched_thresholds'] = {c: {k: mm(ens('iaaft', c, ('matched', k), res, need='matched')) for k in ('tau_hi', 'tau_lo')} for c in 'CA'}
        S['matched_lost6h'] = {c: {kind: mm(ens(kind, c, ('matched', 'lost6h'), res, need='matched')) for kind in ('ft', 'iaaft')} for c in 'CA'}
        w = f'ftw_r0_{res}'
        S['whole_record_gaussian'] = {c: {k: M[w][c][k] for k in ('lost6h', 'N10_ANN', 'half_med_10')} for c in 'CA'} if w in M else None
        S['diagnostics'] = {'real': json.load(open(os.path.join(DATA, f'diag_real_{res}.json')))}
        for kind in ('ft', 'iaaft', 'ftw'):
            ds = [json.load(open(os.path.join(DATA, f))) for f in sorted(os.listdir(DATA)) if f.startswith(f'diag_{kind}_r') and f.endswith(f'_{res}.json')]
            if ds:
                S['diagnostics'][kind] = {k: float(np.mean([d[k] for d in ds])) for k in ('skew', 'neg', 'djf', 'jja', 'acf6h', 'acf24h')}
                if kind == 'iaaft':
                    e = [x[1] for d in ds for x in d['spectral_error']]
                    S['diagnostics']['iaaft_spectral_error_final'] = [float(np.min(e)), float(np.max(e))]
    json.dump(NUM, open(os.path.join(TAB, 'numbers_6h.json'), 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
