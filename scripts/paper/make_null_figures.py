"""
Figures of the comparison with the null fields (daily analysis): concept,
sample fields, counts, geometry and the 1979 step.

    python scripts/paper/make_null_figures.py

Reads only stored results:
    output/experiments/gaussian_null/{metrics.json, rank_test.json}
    output/experiments/gaussian_null_era5/metrics.json
    data/processed_data/experiments/gaussian_null/{feat_*, sample_*}   (histogram, sample fields)
    data/raw/slp.daily.nc                                                (example day)
and writes PDFs to <manuscript>/figures/ (see paths.py).
"""
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GN = os.path.join(ROOT, 'scripts', 'experiments', 'gaussian_null')
sys.path.insert(0, GN)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.pop('GN_SOURCE', None)            # the figures always start from R1
import paths  # noqa: E402
from common import DATA, START_YEAR, build_anomalies, discs, projection_operator  # noqa: E402
from basins import Disc  # noqa: E402
from analyse import load as load_features  # noqa: E402

# The manuscript names the features 1-cyclones and 1-anticyclones and draws its own trajectory figures
# (make_figures.py); HYBRID = False gives the neutral naming and two further summary figures.
HYBRID = True
FIG = paths.FIGURES
OUT_R1 = os.path.join(ROOT, 'output', 'experiments', 'gaussian_null')
OUT_E5 = os.path.join(ROOT, 'output', 'experiments', 'gaussian_null_era5')

plt.rcParams.update({'font.size': 8.5, 'axes.titlesize': 9, 'axes.labelsize': 8.5, 'legend.fontsize': 7,
                     'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False, 'savefig.bbox': 'tight',
                     'savefig.pad_inches': 0.03, 'mathtext.default': 'regular'})
COL = {'real': 'k', 'ft': '#1f77b4', 'iaaft': '#d95f02'}
LAB = {'real': 'reanalysis', 'ft': 'Gaussian null', 'iaaft': 'amplitude-adjusted null'}
SIGN = {'C': '1-cyclones', 'A': '1-anticyclones'} if HYBRID else {'C': 'lows', 'A': 'highs'}
TAUS = [2, 3, 5, 7.5, 10, 15, 20, 25, 30, 40]
C_PIX, W = 86, 172

M1 = json.load(open(os.path.join(OUT_R1, 'metrics.json')))
AN1, M1 = M1['annual'], M1['metrics']
M5 = json.load(open(os.path.join(OUT_E5, 'metrics.json')))['metrics']
RT = json.load(open(os.path.join(OUT_R1, 'rank_test.json')))
SRC = {'R1': (M1, 'real_1979_2023', '_1979_2023_r'), 'ERA5': (M5, 'real_1979_2022', '_1979_2022_r')}


def ens(src, kind):
    M, _, pre = SRC[src]
    return [M[n] for n in sorted(M) if n.startswith(kind + pre)]


def panel_label(ax, s, x=-0.02, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontweight='bold', fontsize=9.5, va='bottom', ha='right')


# ---------------------------------------------------------------- map helpers
def ll2xy(lat, lon):
    """(col, row) pixel coordinates on the 172 x 172 azimuthal-equidistant disc."""
    r = C_PIX * (90 - np.asarray(lat, float)) / 90
    th = np.radians(np.asarray(lon, float) - 90)
    return C_PIX + r * np.cos(th), C_PIX + r * np.sin(th)


def coastlines():
    import shapefile
    try:                                     # Natural Earth 1:110m coastline; cartopy downloads it on first use
        import cartopy.io.shapereader as shpreader
        path = shpreader.natural_earth(resolution='110m', category='physical', name='coastline')
    except Exception as err:
        print('coastlines not drawn:', err)
        return []
    segs = []
    for sh in shapefile.Reader(path).shapes():
        pts = np.array(sh.points)
        parts = list(sh.parts) + [len(pts)]
        for a, b in zip(parts[:-1], parts[1:]):
            p = pts[a:b]
            k = p[:, 1] >= 0
            # split where the line leaves the hemisphere
            idx = np.where(np.diff(k.astype(int)) != 0)[0] + 1
            for q, kk in zip(np.split(p, idx), np.split(k, idx)):
                if kk.all() and len(q) > 1:
                    segs.append(ll2xy(q[:, 1], q[:, 0]))
    return segs


COAST = coastlines()
HALO = [pe.withStroke(linewidth=1.6, foreground='w')]


def draw_disc(ax, field_flat, P, valid, shape, vmax=30):
    sub, _ = discs(field_flat, P, valid, shape)
    g = np.where(sub == sub[0, 0], np.nan, sub) / 100
    im = ax.imshow(g, cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='nearest')
    for x, y in COAST:
        ax.plot(x, y, color='0.25', lw=0.35)
    th = np.linspace(0, 360, 361)
    for lat in (30, 60):
        x, y = ll2xy(np.full_like(th, lat), th)
        ax.plot(x, y, color='0.5', lw=0.3, ls=':')
    x, y = ll2xy(np.zeros_like(th), th)
    ax.plot(x, y, color='0.3', lw=0.5)
    # the stored disc has longitude increasing clockwise in image coordinates; drawing it with the
    # row axis pointing up gives the usual north-polar view (0 deg at the bottom, 90 deg E on the right)
    ax.set_xlim(-2, W + 1); ax.set_ylim(-2, W + 1)
    ax.set_aspect('equal'); ax.axis('off')
    return im


def deep_features(field_flat, P, valid, shape, tau_pa=1000.0):
    """Features of depth >= tau for both signs: dict sign -> (rows, cols, depth_hPa, level, g)."""
    sub, sup = discs(field_flat, P, valid, shape)
    out = {}
    for c, g in (('C', sup), ('A', sub)):
        D = Disc(g)
        F = D.features()
        depth = F['peak'] - F['saddle']
        k = depth >= tau_pa
        rows, cols = np.divmod(F['peak_idx'][k], W)
        out[c] = dict(row=rows, col=cols, depth=depth[k] / 100, peak=F['peak'][k], saddle=F['saddle'][k],
                      parent=F['parent_idx'][k], g=g)
    return out


def half_depth_mask(g, row, col, level):
    sea = g == g[0, 0]
    lab, _ = ndimage.label((g > level) & ~sea, structure=np.ones((3, 3)))
    return lab == lab[row, col]


# ---------------------------------------------------------------- Fig. 1: concept
def fig_concept(anom, P, valid, shape):
    fig = plt.figure(figsize=(7.2, 3.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.08)
    ax = fig.add_subplot(gs[0])
    x = np.linspace(0, 10, 2001)
    f = (-26 * np.exp(-((x - 3.0) / 0.95) ** 2) - 15 * np.exp(-((x - 5.1) / 0.6) ** 2)
         - 11 * np.exp(-((x - 8.1) / 0.7) ** 2) + 5 * np.exp(-((x - 6.7) / 0.5) ** 2) + 2.0 * np.sin(1.3 * x))
    ax.plot(x, f, color='k', lw=1.2)
    ax.axhline(0, color='0.6', lw=0.6, ls='--')
    # minima and saddles of the 1-D profile
    mins = [i for i in range(1, len(x) - 1) if f[i] < f[i - 1] and f[i] < f[i + 1] and f[i] < -5]
    iA, iB, iC = mins[:3]
    sAB = iA + np.argmax(f[iA:iB])                    # saddle between A and B
    # A: deepest, its region reaches the background (rim) -> saddle is the highest level reached
    sadA = 0.0
    sadB = f[sAB]
    iBC = iB + np.argmax(f[iB:iC])
    sadC = min(f[iBC], 0.0) if f[iBC] < 0 else 0.0
    for i, sad, name, dx in ((iA, sadA, 'A', -0.55), (iB, sadB, 'B', 0.5), (iC, sadC, 'C', 0.55)):
        ax.annotate('', xy=(x[i], f[i]), xytext=(x[i], sad),
                    arrowprops=dict(arrowstyle='<->', color='#b2182b', lw=1.0, shrinkA=0, shrinkB=0))
        ax.plot([x[i] - 0.45, x[i] + 0.45], [sad, sad], color='#b2182b', lw=0.8)
        ax.text(x[i] + dx, f[i] - 1.0, name, ha='center', va='top', fontweight='bold')
    ax.text(x[iA] + 0.12, (f[iA] + sadA) / 2 - 4, 'depth', color='#b2182b', rotation=90, va='center', fontsize=7.5)
    ax.plot(x[sAB], f[sAB], 'o', ms=4, mfc='w', mec='#b2182b')
    ax.annotate('saddle of B\n(B is nested in A)', xy=(x[sAB], f[sAB]), xytext=(4.6, 5.5), fontsize=7,
                ha='center', arrowprops=dict(arrowstyle='-', color='0.4', lw=0.5))
    # half-depth footprint of A
    lev = f[iA] + 0.5 * (sadA - f[iA])
    k = np.where(f < lev)[0]
    k = k[(k >= np.searchsorted(x, 1.0)) & (k <= sAB)]
    run = k[(x[k] > x[iA] - 2.5) & (x[k] < x[iA] + 2.5)]
    ax.fill_between(x[run], f[run], lev, color='#2166ac', alpha=0.25, lw=0)
    ax.plot([x[run[0]], x[run[-1]]], [lev, lev], color='#2166ac', lw=1.0)
    ax.text(x[iA], lev + 0.8, 'half-depth\nfootprint', color='#2166ac', ha='center', va='bottom', fontsize=7)
    ax.text(9.9, 0.6, 'background (disc rim)', color='0.4', ha='right', va='bottom', fontsize=7)
    ax.set_xlabel('distance along a transect (schematic)')
    ax.set_ylabel('SLP anomaly (hPa)')
    ax.set_xticks([]); ax.set_ylim(-33, 9.5); ax.set_xlim(0, 10)
    panel_label(ax, 'a', x=-0.08)

    ax = fig.add_subplot(gs[1])
    slot = (2023 - START_YEAR) * 365 + 364          # 31 December 2023
    ff = anom[slot].ravel()
    im = draw_disc(ax, ff, P, valid, shape)
    feats = deep_features(ff, P, valid, shape)
    for c, mk, colr in (('C', 'v', '#053061'), ('A', '^', '#67001f')):
        F = feats[c]
        for r, cc, pk, sd, d in zip(F['row'], F['col'], F['peak'], F['saddle'], F['depth']):
            m = half_depth_mask(F['g'], r, cc, pk - 0.5 * (pk - sd))
            ax.contour(m.astype(float), levels=[0.5], colors=[colr], linewidths=0.9)
            ax.plot(cc, r, mk, ms=4.5, mfc='w', mec='k', mew=0.8)
            ax.text(cc + 5, r + 5, f'{d:.0f}', fontsize=6.5, color='k', path_effects=HALO)
    cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label('SLP anomaly (hPa)')
    panel_label(ax, 'b', x=0.04)
    fig.savefig(os.path.join(FIG, 'fig_concept.pdf'))
    plt.close(fig)
    return {c: sorted(np.round(feats[c]['depth'], 2).tolist(), reverse=True) for c in 'CA'}


# ---------------------------------------------------------------- Fig. 2: sample fields
def fig_sample_fields(anom, P, valid, shape):
    day = 14
    real = anom[(2000 - START_YEAR) * 365 + day].ravel()
    fields = [('reanalysis', real)]
    for k in ('ft', 'iaaft'):
        fields.append((LAB[k],
                       np.load(os.path.join(DATA, f'sample_{k}_1979_2023_r0.npy'))[day].ravel()))
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.75))
    counts = {}
    for ax, (lab, f), pl in zip(axs, fields, 'abc'):
        im = draw_disc(ax, f, P, valid, shape)
        feats = deep_features(f, P, valid, shape)
        for c, mk, colr in (('C', 'v', '#053061'), ('A', '^', '#67001f')):
            ax.plot(feats[c]['col'], feats[c]['row'], mk, ms=4.2, mfc='w', mec='k', mew=0.8, ls='none')
        counts[lab] = (len(feats['C']['row']), len(feats['A']['row']))
        ax.set_title(lab, fontsize=8)
        panel_label(ax, pl, x=0.08, y=0.9)
    cb = fig.colorbar(im, ax=axs, shrink=0.8, pad=0.015, fraction=0.03)
    cb.set_label('SLP anomaly (hPa)')
    fig.savefig(os.path.join(FIG, 'fig_sample_fields.pdf'))
    plt.close(fig)
    return counts


# ---------------------------------------------------------------- Fig. 3: counts against depth
def fig_counts():
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=True)
    M, real, _ = SRC['R1']
    for j, c in enumerate('CA'):
        ax = axs[0, j]
        ax.plot(TAUS, [M[real][f'N{t:g}_{c}_ANN'] for t in TAUS], color='k', marker='o', ms=3, label=LAB['real'])
        for kind in ('ft', 'iaaft'):
            E = np.array([[m[f'N{t:g}_{c}_ANN'] for t in TAUS] for m in ens('R1', kind)])
            ax.plot(TAUS, E.mean(0), color=COL[kind], marker='s', ms=2.5, label=LAB[kind])
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_title(SIGN[c])
        ax.set_ylim(6e-3, 20)
        if j == 0:
            ax.set_ylabel(r'features per day with depth $\geq\tau$')
            ax.legend(frameon=False, loc='lower left')
        panel_label(ax, 'ab'[j], x=-0.07)
        ax = axs[1, j]
        for src, ls in (('R1', '-'), ('ERA5', '--')):
            Ms, rs, _ = SRC[src]
            for kind in ('ft', 'iaaft'):
                R = np.array([[Ms[rs][f'N{t:g}_{c}_ANN'] / m[f'N{t:g}_{c}_ANN'] for t in TAUS] for m in ens(src, kind)])
                ax.fill_between(TAUS, R.min(0), R.max(0), color=COL[kind], alpha=0.2, lw=0)
                ax.plot(TAUS, R.mean(0), ls, color=COL[kind], lw=1.2,
                        label=f'{src}: reanalysis / {LAB[kind]}')
        ax.axhline(1, color='k', lw=0.6)
        ax.set_yscale('log')
        yt = [0.3, 0.5, 0.7, 1, 1.5, 2, 3]
        ax.set_yticks(yt); ax.set_yticklabels([f'{y:g}' for y in yt]); ax.minorticks_off()
        ax.set_ylim(0.2, 3)
        ax.set_xticks(TAUS); ax.set_xticklabels([f'{t:g}' for t in TAUS])
        ax.set_xlabel(r'depth threshold $\tau$ (hPa)')
        if j == 0:
            ax.set_ylabel('ratio to the null')
            ax.legend(frameon=False, loc='lower left', fontsize=6.3)
        panel_label(ax, 'cd'[j], x=-0.07)
    fig.tight_layout(h_pad=1.0, w_pad=1.2)
    fig.savefig(os.path.join(FIG, 'fig_counts.pdf'))
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 4: geometry
def fig_geometry():
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))
    M, real, _ = SRC['R1']
    ax = axs[0]
    for c, mk, ls in (('C', 'v', '-'), ('A', '^', '--')):
        x, y = M[real][f'curve_half_{c}']
        ax.loglog(x, y, color='k', marker=mk, ms=3.5, ls=ls, label=SIGN[c])
        for kind in ('ft', 'iaaft'):
            xs = np.mean([m[f'curve_half_{c}'][0] for m in ens('R1', kind)], 0)
            ys = np.mean([m[f'curve_half_{c}'][1] for m in ens('R1', kind)], 0)
            ax.loglog(xs, ys, color=COL[kind], marker=mk, ms=2.8, ls=ls, lw=0.9)
    for r in (500, 1000):
        a = np.pi * r ** 2 / 1e6
        ax.axhline(a, color='0.6', ls=':', lw=0.7)
        ax.text(*((41, 'right') if r == 500 else (5.1, 'left'))[:1], a * 1.04, f'r = {r} km', fontsize=6, color='0.4',
                ha='right' if r == 500 else 'left')
    ax.set_xticks([5, 10, 20, 40]); ax.set_xticklabels(['5', '10', '20', '40']); ax.minorticks_off()
    ax.set_yticks([0.5, 1, 2, 5]); ax.set_yticklabels(['0.5', '1', '2', '5'])
    ax.set_xlabel('depth (hPa)'); ax.set_ylabel(r'median half-depth area ($10^6$ km$^2$)')
    ax.legend(frameon=False, loc='lower right', handlelength=2.6)
    panel_label(ax, 'a', x=-0.12)

    ax = axs[1]
    bins = np.linspace(100, 2500, 41)
    names = {'ft': sorted(n for n in M1 if n.startswith('ft_1979'))[0],
             'iaaft': sorted(n for n in M1 if n.startswith('iaaft_1979'))[0]}
    for c, tag, ls in (('C', 'sup', '-'), ('A', 'sub', '--')):
        F = load_features('real_1948_2023', tag, (1979, 2023))
        k = F['depth'] >= 10
        ax.hist(np.sqrt(F['half_km2'][k] / np.pi), bins, histtype='step', color='k', ls=ls, density=True, lw=1.1)
        for kind in ('ft', 'iaaft'):
            G = load_features(names[kind], tag)
            k = G['depth'] >= 10
            ax.hist(np.sqrt(G['half_km2'][k] / np.pi), bins, histtype='step', color=COL[kind], ls=ls,
                    density=True, lw=0.8)
    for kind in ('real', 'ft', 'iaaft'):
        ax.plot([], [], color=COL[kind], label=LAB[kind])
    ax.legend(frameon=False, fontsize=6, loc='upper right', handlelength=1.2, borderaxespad=0)
    ax.set_xlabel('equivalent radius (km)')
    ax.set_ylabel('probability density'); ax.set_yticks([])
    panel_label(ax, 'b', x=-0.03)

    ax = axs[2]
    w = 0.13
    for j, c in enumerate('CA'):
        for i, kind in enumerate(('real', 'ft', 'iaaft')):
            h = M[real][f'nest_hist_{c}_10'] if kind == 'real' else np.mean([m[f'nest_hist_{c}_10'] for m in ens('R1', kind)], 0)
            ax.bar(np.arange(4) + (3.4 * j + i - 2.2) * w, np.asarray(h)[:4], w, color=COL[kind],
                   alpha=1 if c == 'C' else 0.45, lw=0)
    ax.set_yscale('log'); ax.set_xticks(range(4))
    ax.set_xlabel('nesting level'); ax.set_ylabel('fraction of features')
    ax.bar([0], [0], color='0.2', label=SIGN['C']); ax.bar([0], [0], color='0.2', alpha=0.45, label=SIGN['A'])
    ax.legend(frameon=False)
    panel_label(ax, 'c', x=-0.12)
    fig.tight_layout(w_pad=0.8)
    fig.savefig(os.path.join(FIG, 'fig_geometry.pdf'))
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 5: track survival
def fig_tracks():
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    L = np.arange(1, 31)
    for ax, c, pl in zip(axs, 'CA', 'ab'):
        for src, ls, lw in (('R1', '-', 1.3), ('ERA5', '--', 1.0)):
            Ms, rs, _ = SRC[src]
            ax.semilogy(L, Ms[rs][f'trk_surv_{c}'], color='k', ls=ls, lw=lw, label=f'{LAB["real"]}, {src}')
            for kind in ('ft', 'iaaft'):
                S = np.array([m[f'trk_surv_{c}'] for m in ens(src, kind)])
                S = np.where(S > 0, S, np.nan)
                if src == 'R1':
                    ax.fill_between(L, np.nanmin(S, 0), np.nanmax(S, 0), color=COL[kind], alpha=0.2, lw=0)
                ax.semilogy(L, np.nanmean(S, 0), color=COL[kind], ls=ls, lw=lw * 0.8, label=f'{LAB[kind]}, {src}')
        ax.set_xlabel(r'track length $\ell$ (days)'); ax.set_title(SIGN[c])
        ax.set_ylim(1e-4, 1.2); ax.set_xlim(1, 28)
        panel_label(ax, pl, x=-0.05)
    axs[0].set_ylabel(r'share of tracks lasting $\geq\ell$ days')
    axs[1].legend(frameon=False, loc='upper right', fontsize=6.3)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(os.path.join(FIG, 'fig_tracks.pdf'))
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 6: summary of ratios
def fig_summary():
    rows = [('N10_{c}_ANN', r'number per day, depth $\geq$ 10 hPa'),
            ('N30_{c}_ANN', r'number per day, depth $\geq$ 30 hPa'),
            ('TP2_{c}', r'second-order total persistence'),
            ('half_med_{c}_10', 'half-depth area (median)'),
            ('mhalf_med_{c}', r'half-depth area at matched count'),
            ('nest_frac_{c}_10', 'share nested'),
            ('nest_ge2_{c}_10', r'share at nesting level $\geq$ 2'),
            ('coh500_{c}', 'next-day coherence within 500 km'),
            ('trk_meanL_{c}', 'mean track length'),
            ('trk_pct_ge5_{c}', r'share of tracks $\geq$ 5 d'),
            ('trk_pct_ge10_{c}', r'share of tracks $\geq$ 10 d'),
            ('trk_median_step_km_{c}', 'median daily step'),
            ('trk_pct1_{c}', 'share of single-day tracks'),
            ('Nall_{c}', 'all extrema per day')]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.9), sharey=True)
    y = np.arange(len(rows))[::-1]
    for ax, c in zip(axs, 'CA'):
        ax.axvspan(0.95, 1.05, color='0.88', lw=0)
        ax.axvline(1, color='k', lw=0.6)
        for i, (key, _) in enumerate(rows):
            k = key.format(c=c)
            for kind, dy in (('ft', 0.17), ('iaaft', -0.17)):
                for src, fc in (('R1', COL[kind]), ('ERA5', 'w')):
                    t = RT[src][k]
                    ax.plot(1 + t[kind]['rel'], y[i] + dy, 'o' if src == 'R1' else 'D', ms=4.2 if src == 'R1' else 3.6,
                            mfc=fc, mec=COL[kind], mew=0.9)
        ax.set_xscale('log')
        xt = [0.5, 0.7, 1, 1.5, 2, 3]
        ax.set_xticks(xt); ax.set_xticklabels([f'{v:g}' for v in xt]); ax.minorticks_off()
        ax.set_xlim(0.42, 3.4)
        ax.set_title(SIGN[c]); ax.set_xlabel('reanalysis / null (ensemble mean)')
        ax.set_yticks(y); ax.set_yticklabels([lab for _, lab in rows])
        ax.tick_params(axis='y', length=0)
        ax.axhline(1.5, color='0.6', lw=0.5, ls=':')
    h = [plt.Line2D([], [], marker='o', ls='none', mfc=COL['ft'], mec=COL['ft'], label='R1, Gaussian null'),
         plt.Line2D([], [], marker='o', ls='none', mfc=COL['iaaft'], mec=COL['iaaft'], label='R1, amplitude-adjusted null'),
         plt.Line2D([], [], marker='D', ls='none', mfc='w', mec=COL['ft'], label='ERA5, Gaussian null'),
         plt.Line2D([], [], marker='D', ls='none', mfc='w', mec=COL['iaaft'], label='ERA5, amplitude-adjusted null')]
    fig.tight_layout(w_pad=0.6, rect=(0, 0.05, 1, 1))
    fig.legend(handles=h, frameon=False, loc='lower center', ncol=4, fontsize=6.8, handletextpad=0.2)
    fig.savefig(os.path.join(FIG, 'fig_summary.pdf'))
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 7: the 1979 step
def fig_step():
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6))
    yrs = np.arange(START_YEAR, 2024)
    for ax, key, c, pl in zip(axs, ('TP2_C', 'TP2_A'), 'CA', 'ab'):
        for kind in ('ft', 'iaaft'):
            for per, y0 in (('1948_1978', 1948), ('1979_2023', 1979)):
                names = sorted(k for k in AN1 if k.startswith(f'{kind}_{per}'))[:4]
                for n in names:
                    v = AN1[n][key]
                    ax.plot(np.arange(y0, y0 + len(v)), v, color=COL[kind], lw=0.5, alpha=0.7)
            ax.plot([], [], color=COL[kind], label=LAB[kind])
        v = np.array(AN1['real_1948_2023'][key])
        ax.plot(yrs, v, color='k', lw=1.1, label=LAB['real'])
        for a, b in ((1948, 1978), (1979, 2023)):
            k = (yrs >= a) & (yrs <= b)
            ax.plot([a, b], [v[k].mean()] * 2, color='k', lw=0.7, ls='--')
        ax.axvline(1978.5, color='0.5', ls=':', lw=0.8)
        ax.set_title(SIGN[c]); ax.set_xlabel('year')
        panel_label(ax, pl, x=-0.08)
    axs[0].set_ylabel(r'annual mean TP$_2$ (hPa$^2$)')
    fig.tight_layout(w_pad=1.2, rect=(0, 0.07, 1, 1))
    hh, ll = axs[0].get_legend_handles_labels()
    fig.legend(hh, ll, frameon=False, loc='lower center', ncol=3)
    fig.savefig(os.path.join(FIG, 'fig_step1979.pdf'))
    plt.close(fig)


def main():
    P, valid, shape = projection_operator()
    anom, _, _ = build_anomalies()
    info = {'example_day_depths_hPa': fig_concept(anom, P, valid, shape),
            'sample_field_counts_C_A': fig_sample_fields(anom, P, valid, shape)}
    del anom
    fig_counts()
    fig_geometry()
    if not HYBRID:              # the manuscript draws its own trajectory figure from the lifted tracker
        fig_tracks()
        fig_summary()
    fig_step()
    json.dump(info, open(os.path.join(FIG, 'figure_info.json'), 'w'), indent=1)
    print(json.dumps(info, indent=1))
    print('figures:', sorted(os.path.basename(p) for p in glob.glob(os.path.join(FIG, '*.pdf'))))


if __name__ == '__main__':
    main()
