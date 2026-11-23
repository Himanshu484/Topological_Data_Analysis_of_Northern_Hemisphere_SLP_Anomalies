"""
Tables of the comparison with the null fields (daily analysis), written as
LaTeX fragments so that no number in a table is typed by hand.  The fragments
use the neutral names "lows" and "highs"; make_tables.py turns the ones the
manuscript uses into its final tables, and this module is also the library
that applies the decision rule.

    python scripts/paper/make_null_tables.py

Inputs (all produced by scripts/experiments/gaussian_null/):
    output/experiments/gaussian_null{,_era5}/metrics.json
    output/experiments/gaussian_null{,_era5}/{half_depth_sensitivity,track_sensitivity}.json   (optional)
    data/processed_data/experiments/gaussian_null{,_era5}/diag_*.json
Outputs: output/paper/null_tables/*.tex and numbers.json (every value, for checking the text).

Decision rule (Section "Test statistics and decision rule" of the paper): a
difference is marked as established if, in both reanalyses and against both
nulls, the reanalysis value lies outside every surrogate realisation, has the
same sign, and differs from the ensemble mean by more than the tolerance
(5 %; 10 % for the day-to-day metrics).
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GN = os.path.join(ROOT, 'scripts', 'experiments', 'gaussian_null')
sys.path.insert(0, GN)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.pop('GN_SOURCE', None)
import paths  # noqa: E402
TAB = paths.NULL_TABLES
OUT = {'R1': os.path.join(ROOT, 'output', 'experiments', 'gaussian_null'),
       'ERA5': os.path.join(ROOT, 'output', 'experiments', 'gaussian_null_era5')}
DAT = {'R1': os.path.join(ROOT, 'data', 'processed_data', 'experiments', 'gaussian_null'),
       'ERA5': os.path.join(ROOT, 'data', 'processed_data', 'experiments', 'gaussian_null_era5')}
REAL = {'R1': 'real_1979_2023', 'ERA5': 'real_1979_2022'}
PER = {'R1': '1979_2023', 'ERA5': '1979_2022'}
M = {s: json.load(open(os.path.join(OUT[s], 'metrics.json')))['metrics'] for s in OUT}
for s in M:                                   # derived low/high ratios
    for n in M[s]:
        for k in ('TP1', 'TP2'):
            M[s][n][f'ratio_{k}'] = M[s][n][f'{k}_C'] / M[s][n][f'{k}_A']
        for k in ('N10', 'N20'):
            M[s][n][f'ratio_{k}'] = M[s][n][f'{k}_C_ANN'] / M[s][n][f'{k}_A_ANN']
NUM = {}
TEMPORAL = ('coh', 'trk_', 'mtrk_', 'mcoh')


def ens(src, kind, key):
    pre = f'{kind}_{PER[src]}_r'
    return np.array([M[src][n][key] for n in sorted(M[src]) if n.startswith(pre)], float)


def stat(src, key):
    r = M[src][REAL[src]][key]
    d = {'real': r}
    for kind in ('ft', 'iaaft'):
        e = ens(src, kind, key)
        d[kind] = dict(mean=e.mean(), min=e.min(), max=e.max(), n=len(e), ratio=r / e.mean(),
                       outside=bool(r > e.max() or r < e.min()))
    return d


TRK = {s: json.load(open(os.path.join(OUT[s], 'track_sensitivity.json'))) for s in OUT
       if os.path.exists(os.path.join(OUT[s], 'track_sensitivity.json'))}
TRK_KEY = {'trk_pct_ge5': 'pct_ge5', 'trk_pct_ge10': 'pct_ge10', 'mtrk_pct_ge10': 'pct_ge10', 'trk_meanL': 'meanL',
           'trk_median_step_km': 'median_step_km', 'trk_pct1': 'pct1'}


def tracker_ratios(key):
    """Reanalysis / null ratios of a track statistic over every tracker setting, source and null (criterion iv)."""
    base, c = key.rsplit('_', 1)
    out = []
    for src, S in TRK.items():
        for cfg, D in S.items():
            for kind in ('ft', 'iaaft'):
                out.append(D['real'][c][TRK_KEY[base]] / np.mean([D[n][c][TRK_KEY[base]] for n in D if n.startswith(kind + '_')]))
    return np.array(out)


def established(key):
    tol = 0.10 if key.startswith(TEMPORAL) else 0.05
    S = [stat(src, key)[kind] for src in ('R1', 'ERA5') for kind in ('ft', 'iaaft')]
    sgn = {np.sign(s['ratio'] - 1) for s in S}
    ok = all(s['outside'] for s in S) and len(sgn) == 1 and all(abs(s['ratio'] - 1) > tol for s in S)
    if ok and key.rsplit('_', 1)[0] in TRK_KEY and TRK:            # criterion (iv): every tracker setting
        r = tracker_ratios(key)
        NUM[f'trkrange_{key}'] = [float(r.min()), float(r.max())]
        ok = len(set(np.sign(r - 1))) == 1 and bool((np.abs(r - 1) > tol).all())
    return ok


def write(name, body):
    open(os.path.join(TAB, name), 'w').write(body)
    print('wrote', name)


ROWS = [('Nall_{c}', r'All extrema per day', '{:.1f}'),
        ('N2_{c}_ANN', r'$N_{2}$ (per day)', '{:.2f}'),
        ('N5_{c}_ANN', r'$N_{5}$', '{:.2f}'),
        ('N10_{c}_ANN', r'$N_{10}$', '{:.2f}'),
        ('N20_{c}_ANN', r'$N_{20}$', '{:.2f}'),
        ('N30_{c}_ANN', r'$N_{30}$', '{:.3f}'),
        ('TP1_{c}', r'$\mathrm{TP}_1$ (hPa)', '{:.1f}'),
        ('TP2_{c}', r'$\mathrm{TP}_2$ (hPa$^2$)', '{:.0f}'),
        ('maxdepth_{c}', r'Largest depth of the day (hPa)', '{:.1f}'),
        None,
        ('half_med_{c}_10', r'Footprint area, median ($10^6$ km$^2$)', '{:.2f}'),
        ('mhalf_med_{c}', r'\quad at matched count', '{:.2f}'),
        ('slope_half_{c}', r'Area--depth exponent', '{:.2f}'),
        ('nest_frac_{c}_10', r'Share nested', '{:.3f}'),
        ('nest_ge2_{c}_10', r'Share at nesting level $\geq 2$', '{:.3f}'),
        None,
        ('coh500_{c}', r'Next-day coherence (500 km)', '{:.3f}'),
        ('trk_meanL_{c}', r'Mean track length (d)', '{:.2f}'),
        ('trk_pct1_{c}', r'Single-day tracks (\%)', '{:.1f}'),
        ('trk_pct_ge5_{c}', r'Tracks $\geq 5$ d (\%)', '{:.1f}'),
        ('trk_pct_ge10_{c}', r'Tracks $\geq 10$ d (\%)', '{:.2f}'),
        ('mtrk_pct_ge10_{c}', r'\quad at matched count', '{:.2f}'),
        ('trk_median_step_km_{c}', r'Median daily step (km)', '{:.0f}')]


def main_table(c):
    lines = [r'\begin{tabular}{@{}lrrrrrcrrrc@{}}', r'\toprule',
             r' & \multicolumn{5}{c}{NCEP--NCAR R1, 1979--2023} & & \multicolumn{3}{c}{ERA5, 1979--2022} & \\',
             r'\cmidrule{2-6}\cmidrule{8-10}',
             r'Property & Rean. & G & AA & R/G & R/AA & & Rean. & R/G & R/AA & Est. \\', r'\midrule']
    for row in ROWS:
        if row is None:
            lines.append(r'\addlinespace')
            continue
        key, lab, fmt = row
        key = key.format(c=c)
        a, b = stat('R1', key), stat('ERA5', key)
        est = established(key)
        NUM[key] = dict(R1=a, ERA5=b, established=est)
        lines.append(' & '.join([lab, fmt.format(a['real']), fmt.format(a['ft']['mean']), fmt.format(a['iaaft']['mean']),
                                 f"{a['ft']['ratio']:.2f}", f"{a['iaaft']['ratio']:.2f}", '',
                                 fmt.format(b['real']), f"{b['ft']['ratio']:.2f}", f"{b['iaaft']['ratio']:.2f}",
                                 r'$\bullet$' if est else '']) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    write(f'tab_main_{c}.tex', '\n'.join(lines) + '\n')


def asymmetry_table():
    lines = [r'\begin{tabular}{@{}lrrrrcrrrr@{}}', r'\toprule',
             r' & \multicolumn{4}{c}{NCEP--NCAR R1} & & \multicolumn{4}{c}{ERA5} \\', r'\cmidrule{2-5}\cmidrule{7-10}',
             r'Low/high ratio of & Rean. & G & AA & Share (\%) & & Rean. & G & AA & Share (\%) \\', r'\midrule']
    for k, lab in (('TP1', r'$\mathrm{TP}_1$'), ('TP2', r'$\mathrm{TP}_2$'), ('N10', r'$N_{10}$'), ('N20', r'$N_{20}$')):
        cells = [lab]
        for src in ('R1', 'ERA5'):
            s = stat(src, f'ratio_{k}')
            share = (np.log(s['iaaft']['mean']) - np.log(s['ft']['mean'])) / (np.log(s['real']) - np.log(s['ft']['mean']))
            NUM[f'ratio_{k}_{src}'] = dict(s, share=share)
            cells += [f"{s['real']:.2f}", f"{s['ft']['mean']:.2f}", f"{s['iaaft']['mean']:.2f}", f'{100 * share:.0f}']
            if src == 'R1':
                cells.append('')
        lines.append(' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    write('tab_asymmetry.tex', '\n'.join(lines) + '\n')


def season_table():
    lines = [r'\begin{tabular}{@{}llrrrrcr@{}}', r'\toprule',
             r' & & \multicolumn{4}{c}{NCEP--NCAR R1} & & ERA5 \\', r'\cmidrule{3-6}\cmidrule{8-8}',
             r' & Season & Rean. & G & AA & R/AA & & R/AA \\', r'\midrule']
    for c, name in (('C', 'Lows'), ('A', 'Highs')):
        for i, s in enumerate(('DJF', 'MAM', 'JJA', 'SON')):
            key = f'N10_{c}_{s}'
            a, b = stat('R1', key), stat('ERA5', key)
            NUM[key] = dict(R1=a, ERA5=b)
            lines.append(' & '.join([name if i == 0 else '', s, f"{a['real']:.2f}", f"{a['ft']['mean']:.2f}",
                                     f"{a['iaaft']['mean']:.2f}", f"{a['iaaft']['ratio']:.2f}", '',
                                     f"{b['iaaft']['ratio']:.2f}"]) + r' \\')
        if c == 'C':
            lines.append(r'\addlinespace')
    lines += [r'\bottomrule', r'\end{tabular}']
    write('tab_seasons.tex', '\n'.join(lines) + '\n')


def diag_table():
    lines = [r'\begin{tabular}{@{}llrrrrrrr@{}}', r'\toprule',
             r' & & Skewness & Area with & \multicolumn{2}{c}{SD (hPa)} & \multicolumn{2}{c}{Autocorrelation} & Zonal corr. \\',
             r' & & & neg.\ skew. & DJF & JJA & lag 1 d & lag 3 d & at $10^\circ$ \\', r'\midrule']
    for src, name in (('R1', 'R1'), ('ERA5', 'ERA5')):
        real = json.load(open(os.path.join(DAT[src], 'diag_real.json')))[PER[src]]
        sets = [('Reanalysis', [real])]
        for kind, lab in (('ft', 'G'), ('iaaft', 'AA')):
            sets.append((lab, [json.load(open(f)) for f in sorted(glob.glob(os.path.join(DAT[src], f'diag_{kind}_{PER[src]}_r*.json')))]))
        for i, (lab, L) in enumerate(sets):
            g = lambda k: float(np.mean([x[k] for x in L]))
            z = float(np.mean([x['zonal_corr_50N_5_10_20deg'][1] for x in L]))
            NUM[f'diag_{src}_{lab}'] = dict(skew=g('skew_20_90'), neg=g('frac_neg_skew'), djf=g('sd_djf_hPa'), jja=g('sd_jja_hPa'),
                                            acf1=g('acf1'), acf3=g('acf3'), zonal10=z, n=len(L))
            if lab == 'AA':
                e = np.array([np.array(x['spectral_error'])[:, 1] for x in L])
                NUM[f'diag_{src}_AA']['spec_err'] = [float(e.mean()), float(e.min()), float(e.max())]
            lines.append(' & '.join([name if i == 0 else '', lab, f"{g('skew_20_90'):+.3f}", f"{g('frac_neg_skew'):.2f}",
                                     f"{g('sd_djf_hPa'):.2f}", f"{g('sd_jja_hPa'):.2f}", f"{g('acf1'):.3f}",
                                     f"{g('acf3'):.3f}", f'{z:.3f}']) + r' \\')
        if src == 'R1':
            lines.append(r'\addlinespace')
    lines += [r'\bottomrule', r'\end{tabular}']
    write('tab_diagnostics.tex', '\n'.join(lines) + '\n')


def step_table():
    AN = json.load(open(os.path.join(OUT['R1'], 'metrics.json')))['annual']
    v0 = 'real_1948_2023'
    sets = {k: {p: sorted(n for n in AN if n.startswith(f'{k}_{p}')) for p in ('1948', '1979')} for k in ('ft', 'iaaft')}
    hyb = sorted(n for n in AN if n.startswith('hyb_1948'))
    lines = [r'\begin{tabular}{@{}lrrrrrr@{}}', r'\toprule',
             r' & \multicolumn{2}{c}{Reanalysis mean} & \multicolumn{4}{c}{Step, 1979--2023 against 1948--78 (\%)} \\',
             r'\cmidrule(r){2-3}\cmidrule(l){4-7}',
             r'Quantity & 1948--78 & 1979--2023 & Reanalysis & G & AA & Hybrid \\', r'\midrule']
    for row in (('TP2_C', r'$\mathrm{TP}_2$, lows (hPa$^2$)', '{:.0f}'), ('TP1_C', r'$\mathrm{TP}_1$, lows (hPa)', '{:.1f}'),
                ('max_C', 'Largest depth, lows (hPa)', '{:.1f}'), ('N10_C', r'$N_{10}$, lows', '{:.2f}'),
                ('N_C', 'All extrema, lows', '{:.1f}'), None,
                ('TP2_A', r'$\mathrm{TP}_2$, highs (hPa$^2$)', '{:.0f}'), ('max_A', 'Largest depth, highs (hPa)', '{:.1f}'),
                ('N10_A', r'$N_{10}$, highs', '{:.2f}'), ('N_A', 'All extrema, highs', '{:.1f}')):
        if row is None:
            lines.append(r'\addlinespace')
            continue
        key, lab, fmt = row
        v = np.array(AN[v0][key])
        r1, r2 = v[:31].mean(), v[31:].mean()
        st = 100 * (r2 / r1 - 1)
        ci = 1.96 * 100 * np.sqrt(v[:31].var(ddof=1) / 31 + v[31:].var(ddof=1) / 45) / r1
        cells = [lab, fmt.format(r1), fmt.format(r2), f'${st:+.1f} \\pm {ci:.1f}$']
        rec = dict(m48=r1, m79=r2, step=st, ci95=ci)
        for kind in ('ft', 'iaaft'):
            m1 = np.array([np.mean(AN[x][key]) for x in sets[kind]['1948']])
            m2 = np.array([np.mean(AN[x][key]) for x in sets[kind]['1979']])
            s = 100 * (m2.mean() / m1.mean() - 1)
            sd = 100 * np.sqrt((m2.std(ddof=1) / m1.mean()) ** 2 + (m2.mean() * m1.std(ddof=1) / m1.mean() ** 2) ** 2)
            rec[kind] = dict(step=s, sd=sd, n48=len(m1), n79=len(m2), z=(st - s) / np.hypot(ci / 1.96, sd))
            cells.append(f'${s:+.1f}$')
        h = 100 * (np.mean([np.mean(AN[x][key]) for x in hyb]) / np.mean([np.mean(AN[x][key]) for x in sets['ft']['1948']]) - 1)
        rec['hybrid'] = h
        cells.append(f'${h:+.1f}$')
        NUM[f'step_{key}'] = rec
        lines.append(' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    write('tab_step.tex', '\n'.join(lines) + '\n')


def sensitivity_tables():
    # tracks: real / null ratios against linking radius and thresholds (R1)
    p = os.path.join(OUT['R1'], 'track_sensitivity.json')
    if os.path.exists(p):
        S = json.load(open(p))
        lines = [r'\begin{tabular}{@{}lrrrrcrrrr@{}}', r'\toprule',
                 r' & \multicolumn{4}{c}{Lows} & & \multicolumn{4}{c}{Highs} \\', r'\cmidrule{2-5}\cmidrule{7-10}',
                 r'$\tau_{\mathrm{lo}}/\tau_{\mathrm{hi}}$ (hPa), $R$ (km) & 1 d (\%) & $\geq 5$ d & $\geq 10$ d & step & & 1 d (\%) & $\geq 5$ d & $\geq 10$ d & step \\',
                 r'\midrule']
        order = ['5/10/500', '5/10/750', '5/10/1000', '5/10/1250', '5/10/1500', None, '10/10/1000', '3/10/1000',
                 '7.5/15/1000', '10/20/1000']
        for cfg in order:
            if cfg is None:
                lines.append(r'\addlinespace')
                continue
            D = S[cfg]
            lo, hi, R = cfg.split('/')
            cells = [f'{lo}/{hi}, {R}']
            rec = {}
            for c in 'CA':
                real = D['real'][c]
                nul = {k: np.mean([D[n][c][k] for n in D if n.startswith('iaaft')]) for k in real}
                gau = {k: np.mean([D[n][c][k] for n in D if n.startswith('ft')]) for k in real}
                rec[c] = dict(real=real, iaaft=nul, ft=gau, n=sum(n.startswith('iaaft') for n in D))
                cells += [f"{real['pct1']:.1f}", f"{real['pct_ge5'] / nul['pct_ge5']:.2f}",
                          f"{real['pct_ge10'] / nul['pct_ge10']:.2f}", f"{real['median_step_km'] / nul['median_step_km']:.2f}"]
                if c == 'C':
                    cells.append('')
            NUM[f'trk_sens_{cfg}'] = rec
            lines.append(' & '.join(cells) + r' \\')
        lines += [r'\bottomrule', r'\end{tabular}']
        write('tab_track_sensitivity.tex', '\n'.join(lines) + '\n')
    # footprint level fraction
    p = os.path.join(OUT['R1'], 'half_depth_sensitivity.json')
    if os.path.exists(p):
        S = json.load(open(p))
        lines = [r'\begin{tabular}{@{}lrrrrcrrrr@{}}', r'\toprule',
                 r' & \multicolumn{4}{c}{Lows} & & \multicolumn{4}{c}{Highs} \\', r'\cmidrule{2-5}\cmidrule{7-10}',
                 r'Level fraction $\phi$ & Rean. & $r_{\mathrm{eq}}$ (km) & R/G & R/AA & & Rean. & $r_{\mathrm{eq}}$ (km) & R/G & R/AA \\',
                 r'\midrule']
        for f in ('1/3', '1/2', '2/3'):
            cells = [f'${f}$']
            rec = {}
            for c in 'CA':
                r = S['real'][c][f]
                g = np.mean([S[n][c][f]['median_1e6km2'] for n in S if n.startswith('ft_')])
                a = np.mean([S[n][c][f]['median_1e6km2'] for n in S if n.startswith('iaaft_')])
                rec[c] = dict(real=r['median_1e6km2'], req=r['req_km'], ft=g, iaaft=a,
                              n=sum(n.startswith('iaaft_') for n in S))
                cells += [f"{r['median_1e6km2']:.2f}", f"{r['req_km']:.0f}", f"{r['median_1e6km2'] / g:.2f}",
                          f"{r['median_1e6km2'] / a:.2f}"]
                if c == 'C':
                    cells.append('')
            NUM[f'half_sens_{f}'] = rec
            lines.append(' & '.join(cells) + r' \\')
        lines += [r'\bottomrule', r'\end{tabular}']
        write('tab_footprint_sensitivity.tex', '\n'.join(lines) + '\n')


def coherence_table(radius='500', lags=(1, 2, 3, 5, 7)):
    L = {s: json.load(open(os.path.join(OUT[s], 'lagged_coherence.json'))) for s in OUT}
    all_lags = L['R1']['_lags']
    assert all_lags[-1] == 365

    def series(src, c):
        D = L[src]
        real = np.array(D['real'][c][radius])
        nul = {k: np.mean([D[n][c][radius] for n in D if n.startswith(k + '_')], 0) for k in ('ft', 'iaaft')}
        return real, nul

    lines = [r'\begin{tabular}{@{}lrrrrcrrrr@{}}', r'\toprule',
             r' & \multicolumn{4}{c}{Lows} & & \multicolumn{4}{c}{Highs} \\', r'\cmidrule{2-5}\cmidrule{7-10}',
             r' & \multicolumn{2}{c}{$c_k$, R1} & \multicolumn{2}{c}{Excess ratio} & & \multicolumn{2}{c}{$c_k$, R1} & \multicolumn{2}{c}{Excess ratio} \\',
             r'Lag $k$ (d) & Rean. & AA & R1 & ERA5 & & Rean. & AA & R1 & ERA5 \\', r'\midrule']
    rec = {}
    for k in list(lags) + [365]:
        i = all_lags.index(k)
        cells = [str(k) if k != 365 else r'365 (chance)']
        for c in 'CA':
            real, nul = series('R1', c)
            cells += [f'{real[i]:.3f}', f"{nul['iaaft'][i]:.3f}"]
            for src in ('R1', 'ERA5'):
                r_, n_ = series(src, c)
                ex = {kind: (r_[i] - r_[-1]) / (n_[kind][i] - n_[kind][-1]) for kind in ('ft', 'iaaft')} if k != 365 else None
                rec[f'{src}_{c}_{k}'] = dict(real=float(r_[i]), aa=float(n_['iaaft'][i]), g=float(n_['ft'][i]), excess_ratio=ex)
                cells.append(f"{ex['iaaft']:.2f}" if ex else '--')
            if c == 'C':
                cells.append('')
        if k == 365:
            lines.append(r'\addlinespace')
        lines.append(' & '.join(cells) + r' \\')
    NUM[f'lagged_coherence_{radius}km'] = rec
    lines += [r'\bottomrule', r'\end{tabular}']
    write('tab_coherence.tex', '\n'.join(lines) + '\n')
    # the same excess ratios within 1000 km, for the text
    rec2 = {}
    for src in ('R1', 'ERA5'):
        for c in 'CA':
            D = L[src]
            real = np.array(D['real'][c]['1000'])
            aa = np.mean([D[n][c]['1000'] for n in D if n.startswith('iaaft_')], 0)
            rec2[f'{src}_{c}'] = {str(k): float((real[all_lags.index(k)] - real[-1]) / (aa[all_lags.index(k)] - aa[-1])) for k in lags}
    NUM['lagged_coherence_1000km_excess_ratio_AA'] = rec2


def main():
    for c in 'CA':
        main_table(c)
    coherence_table()
    asymmetry_table()
    season_table()
    diag_table()
    step_table()
    sensitivity_tables()
    # a few extra in-text numbers
    for src in ('R1', 'ERA5'):
        r = M[src][REAL[src]]
        NUM[f'extra_{src}'] = {k: r[k] for k in ('half_req_km_C_10', 'half_req_km_A_10', 'trk_maxL_C', 'trk_maxL_A',
                                                 'basin_med_C_10', 'basin_med_A_10', 'pole_frac_C_10', 'pole_frac_A_10',
                                                 'half_med_C_5_10', 'half_med_A_5_10', 'half_med_C_20_40', 'half_med_A_20_40',
                                                 'mtau_C', 'mtau_A')}
        for kind in ('ft', 'iaaft'):
            NUM[f'extra_{src}'][f'mtau_{kind}'] = {c: float(ens(src, kind, f'mtau_{c}').mean()) for c in 'CA'}
            NUM[f'extra_{src}'][f'halfbins_{kind}'] = {k: float(ens(src, kind, k).mean()) for k in
                                                       ('half_med_C_5_10', 'half_med_A_5_10', 'half_med_C_20_40', 'half_med_A_20_40')}
    g = M['R1'].get('ftg_1979_2023_r0')
    if g:
        r = M['R1'][REAL['R1']]
        keys = ('trk_pct_ge5_C', 'trk_pct_ge5_A', 'trk_pct_ge10_C', 'trk_pct_ge10_A', 'coh500_C', 'coh500_A',
                'trk_median_step_km_C', 'trk_median_step_km_A', 'trk_meanL_C', 'trk_meanL_A', 'N10_C_ANN', 'N10_A_ANN')
        NUM['seamfree'] = {k: dict(real=r[k], ftg=g[k], ratio=r[k] / g[k]) for k in keys}
    json.dump(NUM, open(os.path.join(TAB, 'numbers.json'), 'w'), indent=1, default=float)
    est = sorted(k for k, v in NUM.items() if isinstance(v, dict) and v.get('established'))
    print('established:', est)
    print('not established:', sorted(k for k, v in NUM.items() if isinstance(v, dict) and v.get('established') is False))


if __name__ == '__main__':
    main()
