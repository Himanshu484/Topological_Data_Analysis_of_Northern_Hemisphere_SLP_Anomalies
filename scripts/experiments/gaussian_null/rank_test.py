"""
Rank test of real vs surrogate ensembles, for R1 (1979-2023) and, if present,
ERA5 (1979-2022).  Reads output/experiments/gaussian_null{,_era5}/metrics.json
(written by analyse.py with and without GN_SOURCE=era5) and writes
output/experiments/gaussian_null/rank_test.md and rank_test.json.

For N surrogates of one kind, the real value's rank gives
    p_hi = (1 + #{s >= real}) / (N + 1),   p_lo = (1 + #{s <= real}) / (N + 1).
R1 is tested two-sided, p = min(1, 2 min(p_hi, p_lo)): with N = 39 the smallest
possible p is 0.05.  ERA5 is a replication, so it is tested one-sided in the
direction R1 found (sign of R1 real - R1 surrogate mean for that kind): with
N = 19 the smallest possible p is 0.05.  Theiler's S = (real - mean) / SD is
reported alongside.  No correction for the number of metrics is applied; the
table flags every row, and the reader should expect ~5 % of truly null rows to
reach p <= 0.05 by chance (rows reach the rank floor only when the real value
lies outside every surrogate).
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
OUTS = {'R1': os.path.join(ROOT, 'output', 'experiments', 'gaussian_null'),
        'ERA5': os.path.join(ROOT, 'output', 'experiments', 'gaussian_null_era5')}
REALS = {'R1': 'real_1979_2023', 'ERA5': 'real_1979_2022'}
PREFIX = {'R1': '_1979_2023_r', 'ERA5': '_1979_2022_r'}

ROWS = [('Nall_{c}', 'all extrema per day', '{:.1f}'),
        ('N2_{c}_ANN', 'N(depth ≥ 2 hPa)/day', '{:.2f}'),
        ('N10_{c}_ANN', 'N(≥ 10)/day', '{:.2f}'),
        ('N30_{c}_ANN', 'N(≥ 30)/day', '{:.3f}'),
        ('N10_{c}_DJF', 'N(≥ 10)/day DJF', '{:.2f}'),
        ('N10_{c}_JJA', 'N(≥ 10)/day JJA', '{:.2f}'),
        ('TP2_{c}', 'TP₂ (hPa²)', '{:.0f}'),
        ('half_med_{c}_10', 'half-depth area ≥10 hPa (10⁶ km²)', '{:.2f}'),
        ('mhalf_med_{c}', '  same at count-matched τ*', '{:.2f}'),
        ('slope_half_{c}', 'slope d log A_half / d log depth', '{:.2f}'),
        ('nest_frac_{c}_10', 'nested, ≥10 hPa', '{:.3f}'),
        ('nest_ge2_{c}_10', 'nesting level ≥ 2, ≥10 hPa', '{:.3f}'),
        ('coh500_{c}', 'next-day coherence 500 km', '{:.3f}'),
        ('trk_meanL_{c}', 'track mean length (d)', '{:.2f}'),
        ('trk_pct1_{c}', 'single-day tracks (%)', '{:.1f}'),
        ('trk_pct_ge5_{c}', 'tracks ≥ 5 d (%)', '{:.1f}'),
        ('trk_pct_ge10_{c}', 'tracks ≥ 10 d (%)', '{:.2f}'),
        ('mtrk_pct_ge10_{c}', '  same at τ*', '{:.2f}'),
        ('trk_median_step_km_{c}', 'median daily step (km)', '{:.0f}')]
RATIOS = [('TP1', 'C/A ratio TP₁'), ('TP2', 'C/A ratio TP₂'), ('N10', 'C/A ratio N(≥10)'), ('N20', 'C/A ratio N(≥20)')]


def load(src):
    p = os.path.join(OUTS[src], 'metrics.json')
    if not os.path.exists(p):
        return None
    M = json.load(open(p))['metrics']
    for n in M:
        for k, _ in RATIOS:
            a, b = (f'{k}_C', f'{k}_A') if k.startswith('TP') else (f'{k}_C_ANN', f'{k}_A_ANN')
            M[n][f'ratio_{k}'] = M[n][a] / M[n][b]
    ens = {kind: sorted((n for n in M if n.startswith(kind + PREFIX[src])), key=lambda s: int(s.split('_r')[-1]))
           for kind in ('ft', 'iaaft')}
    return M, ens


def test(real, sur, direction=None):
    sur = np.asarray(sur, float)
    N = len(sur)
    p_hi = (1 + np.sum(sur >= real)) / (N + 1)
    p_lo = (1 + np.sum(sur <= real)) / (N + 1)
    if direction is None:
        p = min(1.0, 2 * min(p_hi, p_lo))
    else:
        p = p_hi if direction > 0 else p_lo
    sd = sur.std(ddof=1)
    return dict(mean=float(sur.mean()), sd=float(sd), min=float(sur.min()), max=float(sur.max()),
                rel=float((real - sur.mean()) / sur.mean()), S=float((real - sur.mean()) / sd) if sd > 0 else np.inf,
                p=float(p), N=N)


def skew_share(M, ens, real_name, k):
    """Share of log(C/A)_real that the amplitude-adjusted null reproduces beyond the Gaussian one."""
    r = np.log(M[real_name][f'ratio_{k}'])
    f = np.log(np.mean([M[n][f'ratio_{k}'] for n in ens['ft']]))
    a = np.log(np.mean([M[n][f'ratio_{k}'] for n in ens['iaaft']]))
    return float((a - f) / (r - f))


def figure(D):
    """Real / surrogate N(depth >= tau) per day against tau; band = range over realisations."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    taus = [2, 3, 5, 7.5, 10, 15, 20, 25, 30, 40]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    style = {'R1': '-', 'ERA5': '--'}
    col = {'ft': 'tab:blue', 'iaaft': 'tab:orange'}
    for a, c, nm in zip(ax, ('C', 'A'), ('cyclones (lows)', 'anticyclones (highs)')):
        for src, (M, ens) in D.items():
            for kind in ('ft', 'iaaft'):
                R = np.array([[M[REALS[src]][f'N{t:g}_{c}_ANN'] / M[n][f'N{t:g}_{c}_ANN'] for t in taus] for n in ens[kind]])
                a.fill_between(taus, R.min(0), R.max(0), color=col[kind], alpha=0.25, lw=0)
                a.plot(taus, R.mean(0), style[src], color=col[kind], label=f'{src}: real / {kind} (N={len(R)})')
        a.axhline(1, color='k', lw=0.8)
        a.set_xscale('log'); a.set_yscale('log')
        a.set_xticks(taus); a.set_xticklabels([f'{t:g}' for t in taus])
        yt = [0.3, 0.4, 0.5, 0.7, 1, 1.5, 2, 3]
        a.set_yticks(yt); a.set_yticklabels([f'{y:g}' for y in yt]); a.minorticks_off()
        a.grid(alpha=0.3)
        a.set_xlabel('depth threshold τ (hPa)'); a.set_title(nm)
    ax[0].set_ylabel('N(depth ≥ τ) real / surrogate')
    ax[0].legend(fontsize=8)
    fig.suptitle('Real vs surrogate feature counts, annual: R1 1979–2023 and ERA5 1979–2022')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTS['R1'], 'fig_rank_counts_r1_era5.png'), dpi=130)
    plt.close(fig)


def main():
    D = {s: load(s) for s in OUTS}
    D = {s: v for s, v in D.items() if v is not None}
    keys = [(k.format(c=c), f'{"C" if c == "C" else "A"}: {lab}', fmt) for c in ('C', 'A') for k, lab, fmt in ROWS]
    keys += [(f'ratio_{k}', lab, '{:.2f}') for k, lab in RATIOS]
    res = {}
    for src, (M, ens) in D.items():
        res[src] = {}
        for key, lab, fmt in keys:
            real = M[REALS[src]][key]
            res[src][key] = {'real': real}
            for kind in ('ft', 'iaaft'):
                sur = [M[n][key] for n in ens[kind]]
                direction = None
                if src != 'R1':
                    R = res['R1'][key][kind]
                    direction = np.sign(res['R1'][key]['real'] - R['mean'])
                res[src][key][kind] = test(real, sur, direction)
        res[src]['_skew_share'] = {k: skew_share(M, ens, REALS[src], k) for k, _ in RATIOS}
        res[src]['_N'] = {kind: len(ens[kind]) for kind in ens}

    def cell(t, fmt):
        star = '**' if t['p'] <= 0.05 else ''
        return f'{fmt.format(t["mean"])} [{fmt.format(t["min"])}, {fmt.format(t["max"])}]; {100 * t["rel"]:+.0f} %, S {t["S"]:+.0f}, {star}p {t["p"]:.3f}{star}'

    lines = ['# Rank test: real vs surrogate ensembles\n',
             'Surrogate cells: ensemble mean [min, max]; real vs mean in %; Theiler S; rank p-value '
             '(bold if ≤ 0.05). R1 two-sided; ERA5 one-sided in the direction R1 found. '
             'See `rank_test.py` for the definitions.\n']
    for src in D:
        yrs = REALS[src][5:].replace('_', '–')
        lines.append(f'\n## {src} {yrs} (N = {res[src]["_N"]["ft"]} ft, {res[src]["_N"]["iaaft"]} iaaft)\n')
        lines.append('| property | real | Gaussian (ft) | amplitude-adjusted (iaaft) |\n|---|---|---|---|')
        for key, lab, fmt in keys:
            r = res[src][key]
            lines.append(f'| {lab} | {fmt.format(r["real"])} | {cell(r["ft"], fmt)} | {cell(r["iaaft"], fmt)} |')
        sh = res[src]['_skew_share']
        lines.append('\nShare of log(C/A) explained by skewness (iaaft − ft over real − ft): '
                     + ', '.join(f'{lab[10:]} {100 * sh[k]:.0f} %' for k, lab in RATIOS) + '.')
    if len(D) == 2:
        lines.append('\n## Replication summary: real / iaaft-mean, R1 vs ERA5\n')
        lines.append('| property | R1 real | ERA5 real | R1 real/iaaft | ERA5 real/iaaft | R1 p (2-sided) | ERA5 p (1-sided) | replicates? |\n|---|---|---|---|---|---|---|---|')
        for key, lab, fmt in keys:
            a, b = res['R1'][key], res['ERA5'][key]
            same = np.sign(a['iaaft']['rel']) == np.sign(b['iaaft']['rel'])
            rep = 'yes' if (a['iaaft']['p'] <= 0.05 and b['iaaft']['p'] <= 0.05 and same) else (
                  'n/a (null in R1)' if a['iaaft']['p'] > 0.05 else 'no')
            lines.append(f'| {lab} | {fmt.format(a["real"])} | {fmt.format(b["real"])} | {1 + a["iaaft"]["rel"]:.2f} | '
                         f'{1 + b["iaaft"]["rel"]:.2f} | {a["iaaft"]["p"]:.3f} | {b["iaaft"]["p"]:.3f} | {rep} |')
    figure(D)
    open(os.path.join(OUTS['R1'], 'rank_test.md'), 'w').write('\n'.join(lines) + '\n')
    json.dump(res, open(os.path.join(OUTS['R1'], 'rank_test.json'), 'w'), indent=1, default=float)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
