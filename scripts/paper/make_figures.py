"""
Climatology and trajectory figures of the manuscript.

    python scripts/paper/trajectory_analysis.py; GN_SOURCE=era5 python scripts/paper/trajectory_analysis.py
    python scripts/paper/make_null_figures.py      # concept, sample fields, counts, geometry, 1979 step
    python scripts/paper/make_figures.py           # this file

Writes to <manuscript>/figures/ (see paths.py):
    fig_h_tp_daily.pdf          daily climatology of total persistence, 1979-2023, with 95 % bands
    fig_h_tp_monthly.pdf        monthly series of total persistence
    fig_h_traj_monthly.pdf      trajectory statistics by starting month, with 95 % bands
    fig_h_tracking_example.pdf  three consecutive days: maps and persistence diagrams with tracked features
    fig_h_event2012.pdf         the high of January-February 2012
    fig_h_tracks_null.pdf       track-length distribution against the nulls
"""
import json
import os
import sys

import numpy as np

os.environ.pop('GN_SOURCE', None)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_null_figures as B  # noqa: E402  (shared map helpers, colours, paths)
from make_null_figures import plt, ROOT, FIG, START_YEAR, panel_label  # noqa: E402
from analyse import load  # noqa: E402
from common import MAIN_PERIOD, REAL, build_anomalies, discs, projection_operator  # noqa: E402
from basins import Disc  # noqa: E402
from tracking import pix2ll  # noqa: E402

OUT = os.path.join(ROOT, 'output', 'paper')
DAT = os.path.join(ROOT, 'data', 'processed_data', 'paper')
A1 = json.load(open(os.path.join(OUT, 'analysis_R1.json')))
A5 = json.load(open(os.path.join(OUT, 'analysis_ERA5.json')))
CC = {'C': '#2166ac', 'A': '#b2182b'}
NAME = {'C': '1-cyclones', 'A': '1-anticyclones'}
MONTHS = list('JFMAMJJASOND')
MONTH_LEN = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
MSTART = np.concatenate([[0], np.cumsum(MONTH_LEN)[:-1]])
W = 172


def slot_of(year, month, day):
    return (year - START_YEAR) * 365 + int(MSTART[month - 1]) + day - 1


def date_label(slot):
    d = slot % 365
    m = int(np.searchsorted(np.cumsum(MONTH_LEN), d, 'right'))
    return f"{d - MSTART[m] + 1} {['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][m]} {START_YEAR + slot // 365}"


def aggregates():
    ny = MAIN_PERIOD[1] - MAIN_PERIOD[0] + 1
    out = {}
    for c, tag in (('C', 'sup'), ('A', 'sub')):
        agg = load(REAL, tag, MAIN_PERIOD)['agg'].reshape(ny, 365, 4)
        out[c] = {'TP1': agg[..., 1] / 100, 'TP2': agg[..., 2] / 1e4}
    return out


def fig_tp_daily(G):
    rng = np.random.default_rng(1)
    fig, axs = plt.subplots(2, 1, figsize=(6.2, 5.0), sharex=True)
    for ax, k, lab, pl in zip(axs, ('TP1', 'TP2'), (r'TP$_1$ (hPa)', r'TP$_2$ (hPa$^2$)'), 'ab'):
        for c, ls in (('C', '-'), ('A', '--')):
            v = G[c][k]
            ny = v.shape[0]
            bs = np.array([v[rng.integers(0, ny, ny)].mean(0) for _ in range(500)])
            lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
            ax.fill_between(np.arange(365), lo, hi, color=CC[c], alpha=0.25, lw=0)
            ax.plot(np.arange(365), v.mean(0), ls, color=CC[c], lw=1.0, label=NAME[c])
        ax.set_ylabel(lab)
        ax.set_xlim(0, 364)
        panel_label(ax, pl, x=-0.07)
    axs[1].set_xticks(MSTART + 15)
    axs[1].set_xticklabels(MONTHS)
    axs[1].tick_params(axis='x', length=0)
    axs[0].legend(frameon=False, loc='lower left')
    fig.tight_layout(h_pad=0.6)
    fig.savefig(os.path.join(FIG, 'fig_h_tp_daily.pdf'))
    plt.close(fig)


def fig_tp_monthly(G):
    month_of_doy = np.repeat(np.arange(12), MONTH_LEN)
    fig, axs = plt.subplots(2, 1, figsize=(6.8, 4.6), sharex=True)
    for ax, k, lab, pl in zip(axs, ('TP1', 'TP2'), (r'monthly mean TP$_1$ (hPa)', r'monthly mean TP$_2$ (hPa$^2$)'), 'ab'):
        for c, ls in (('C', '-'), ('A', '--')):
            v = G[c][k]
            mm = np.stack([v[:, month_of_doy == m].mean(1) for m in range(12)], 1).ravel()
            t = MAIN_PERIOD[0] + (np.arange(len(mm)) + 0.5) / 12
            ax.plot(t, mm, ls, color=CC[c], lw=0.7, label=NAME[c])
        ax.set_ylabel(lab)
        ax.set_xlim(MAIN_PERIOD[0], MAIN_PERIOD[1] + 1)
        panel_label(ax, pl, x=-0.07)
    axs[1].set_xlabel('year')
    axs[0].legend(frameon=False, loc='upper right', ncol=2)
    fig.tight_layout(h_pad=0.6)
    fig.savefig(os.path.join(FIG, 'fig_h_tp_monthly.pdf'))
    plt.close(fig)


def fig_traj_monthly():
    rows = [('mean_length', 'mean length (days)'), ('count_per_year', 'tracks started per 30 days'),
            ('mean_depth', 'mean depth (hPa)'), ('mean_area', r'mean footprint ($10^6$ km$^2$)')]
    fig, axs = plt.subplots(4, 2, figsize=(6.6, 7.4), sharex=True)
    x = np.arange(12)
    for j, c in enumerate('CA'):
        for i, (k, lab) in enumerate(rows):
            ax = axs[i, j]
            m = A1[c]['monthly'][k]
            f = 30.0 / np.array(MONTH_LEN) if k == 'count_per_year' else 1.0     # remove the length of the month
            ax.fill_between(x, f * np.array(m['lo']), f * np.array(m['hi']), color=CC[c], alpha=0.25, lw=0)
            ax.plot(x, f * np.array(m['value']), '-o', color=CC[c], ms=3, lw=1.0, label='NCEP-NCAR R1')
            ax.plot(x, f * np.array(A5[c]['monthly'][k]['value']), '--', color='0.35', lw=0.9, label='ERA5')
            if j == 0:
                ax.set_ylabel(lab)
            if i == 0:
                ax.set_title(NAME[c])
            panel_label(ax, 'abcdefgh'[2 * i + j], x=-0.1, y=1.0)
        axs[3, j].set_xticks(x)
        axs[3, j].set_xticklabels(MONTHS)
        axs[3, j].set_xlabel('month in which the track starts')
    for i in range(4):                       # common vertical scale within a row, except the counts
        if i != 1:
            lo = min(axs[i, 0].get_ylim()[0], axs[i, 1].get_ylim()[0])
            hi = max(axs[i, 0].get_ylim()[1], axs[i, 1].get_ylim()[1])
            for j in range(2):
                axs[i, j].set_ylim(lo, hi)
    axs[0, 1].legend(frameon=False, loc='upper right')
    fig.tight_layout(h_pad=0.5, w_pad=1.0)
    fig.savefig(os.path.join(FIG, 'fig_h_traj_monthly.pdf'))
    plt.close(fig)


def day_features(field_flat, P, valid, shape, c):
    """All features of one sign on one day: anomaly at birth (saddle) and death (extremum), position, level set."""
    sub, sup = discs(field_flat, P, valid, shape)
    g = sup if c == 'C' else sub
    F = Disc(g).features()
    sgn = -1.0 if c == 'C' else 1.0
    lat, lon = pix2ll(*np.divmod(F['peak_idx'], W))
    return dict(birth=sgn * F['saddle'] / 100, death=sgn * F['peak'] / 100, depth=(F['peak'] - F['saddle']) / 100,
                row=F['peak_idx'] // W, col=F['peak_idx'] % W, lat=lat, lon=lon, peak=F['peak'], saddle=F['saddle'], g=g)


def locate(D, lat, lon):
    k = np.where(np.isclose(D['lat'], lat, atol=1e-6) & np.isclose(D['lon'], lon, atol=1e-6))[0]
    assert len(k) == 1, (lat, lon, k)
    return int(k[0])


def footprint(ax, D, i, color, lw=1.1):
    m = B.half_depth_mask(D['g'], D['row'][i], D['col'][i], D['peak'][i] - 0.5 * (D['peak'][i] - D['saddle'][i]))
    ax.contour(m.astype(float), levels=[0.5], colors=[color], linewidths=lw)


def fig_tracking_example(anom, P, valid, shape):
    Z = np.load(os.path.join(DAT, 'tracks_R1_C.npz'))
    slots = [slot_of(2012, 2, d) for d in (2, 3, 4)]
    cand = {}
    for t in np.unique(Z['track'][np.isin(Z['slot'], slots)]):
        m = (Z['track'] == t) & np.isin(Z['slot'], slots)
        if m.sum() == 3:
            cand[int(t)] = float(Z['depth'][m].mean())
    chosen = sorted(cand, key=cand.get, reverse=True)[:2]
    cols = ['#d95f02', '#1b9e77']
    info = []
    fig, axs = plt.subplots(2, 3, figsize=(7.0, 4.9), gridspec_kw=dict(height_ratios=[1.15, 1]))
    for j, s in enumerate(slots):
        f = anom[s].ravel()
        im = B.draw_disc(axs[0, j], f, P, valid, shape)
        D = day_features(f, P, valid, shape, 'C')
        ax = axs[1, j]
        ax.plot(D['birth'], D['death'], '.', color='k', ms=2.5)
        lim = (-42, 12)
        ax.plot(lim, lim, color='0.4', lw=0.7)
        for t, col in zip(chosen, cols):
            m = (Z['track'] == t) & (Z['slot'] == s)
            i = locate(D, Z['lat'][m][0], Z['lon'][m][0])
            footprint(axs[0, j], D, i, col)
            axs[0, j].plot(D['col'][i], D['row'][i], 's', ms=3.5, mfc='#ffd92f', mec='k', mew=0.6)
            ax.plot(D['birth'][i], D['death'][i], '*', ms=11, mfc=col, mec='k', mew=0.5)
            n = int((Z['track'] == t).sum())
            info.append(dict(track=t, date=date_label(s), lat=float(D['lat'][i]), lon=float(D['lon'][i]),
                             depth=float(D['depth'][i]), birth=float(D['birth'][i]), death=float(D['death'][i]), length=n))
        axs[0, j].set_title(date_label(s), fontsize=8.5)
        ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_aspect('equal')
        ax.set_xlabel('birth (hPa)')
        if j == 0:
            ax.set_ylabel('death (hPa)')
    cb = fig.colorbar(im, ax=axs[0, :], shrink=0.85, pad=0.015, fraction=0.025)
    cb.set_label('SLP anomaly (hPa)')
    fig.savefig(os.path.join(FIG, 'fig_h_tracking_example.pdf'))
    plt.close(fig)
    return info


def fig_event2012(anom, P, valid, shape):
    Z = np.load(os.path.join(DAT, 'tracks_R1_A.npz'))
    ev = [t for t in A1['event_2012'] if max(d['depth'] for d in t['days']) >= 40]
    assert len(ev) == 2, [t['track'] for t in A1['event_2012']]
    cols = ['#7570b3', '#d95f02']
    days = [(2012, 1, 23), (2012, 1, 27), (2012, 1, 30), (2012, 1, 31), (2012, 2, 1), (2012, 2, 4)]
    fig = plt.figure(figsize=(7.0, 6.6))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.72], hspace=0.16, wspace=0.04)
    for n, (y, mth, d) in enumerate(days):
        ax = fig.add_subplot(gs[n // 3, n % 3])
        s = slot_of(y, mth, d)
        f = anom[s].ravel()
        im = B.draw_disc(ax, f, P, valid, shape, vmax=45)
        D = day_features(f, P, valid, shape, 'A')
        for t, col in zip(ev, cols):
            m = Z['track'] == t['track']
            o = np.argsort(Z['slot'][m])
            sl, la, lo = Z['slot'][m][o], Z['lat'][m][o], Z['lon'][m][o]
            past = sl <= s
            if past.any():
                x, yy = B.ll2xy(la[past], lo[past])
                ax.plot(x, yy, '-', color=col, lw=1.0)
            if (sl == s).any():
                i = locate(D, la[sl == s][0], lo[sl == s][0])
                footprint(ax, D, i, col, lw=1.3)
                ax.plot(D['col'][i], D['row'][i], 's', ms=4, mfc='#ffd92f', mec='k', mew=0.6)
                ax.text(D['col'][i] + 4, D['row'][i] + 4, f"{D['depth'][i]:.0f}", fontsize=7, path_effects=B.HALO)
        ax.set_xlim(52, 142); ax.set_ylim(22, 112)
        ax.set_title(date_label(s), fontsize=8.5, pad=2)
    cb = fig.colorbar(im, ax=fig.axes[:6], shrink=0.8, pad=0.015, fraction=0.025)
    cb.set_label('SLP anomaly (hPa)')
    ax = fig.add_subplot(gs[2, :])
    s0 = slot_of(2012, 1, 1)
    for t, col, lab in zip(ev, cols, ('first track', 'second track')):
        x = [slot_of(*map(int, d['date'].split('-'))) - s0 + 1 for d in t['days']]
        ax.plot(x, [d['depth'] for d in t['days']], '-o', color=col, ms=3.5, label=f'depth, {lab}')
        ax.plot(x, [d['anom'] for d in t['days']], ':', color=col, lw=1.0)
    ax.plot([], [], ':', color='0.3', label='anomaly at the maximum')
    ax.set_xticks([21, 24, 27, 30, 33, 36, 38])
    ax.set_xticklabels(['21 Jan', '24 Jan', '27 Jan', '30 Jan', '2 Feb', '5 Feb', '7 Feb'])
    ax.set_ylabel('hPa'); ax.set_ylim(0, 52)
    ax.legend(frameon=False, loc='lower right', ncol=3, fontsize=6.5)
    fig.savefig(os.path.join(FIG, 'fig_h_event2012.pdf'))
    plt.close(fig)
    return [dict(track=t['track'], start=t['days'][0]['date'], end=t['days'][-1]['date'], length=t['length']) for t in ev]


def fig_tracks_null():
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    L = np.arange(1, 31)
    for ax, c, pl in zip(axs, 'CA', 'ab'):
        for A, src, ls, lw in ((A1, 'R1', '-', 1.3), (A5, 'ERA5', '--', 1.0)):
            ax.semilogy(L, A[c]['reference']['survival'], color='k', ls=ls, lw=lw, label=f'reanalysis, {src}')
            for kind in ('ft', 'iaaft'):
                S = A['surrogates'][kind][c]['reference']['survival']
                mean = np.array(S['mean']); mean = np.where(mean > 0, mean, np.nan)
                if src == 'R1':
                    lo = np.where(np.array(S['min']) > 0, S['min'], 1e-6)
                    ax.fill_between(L, lo, S['max'], color=B.COL[kind], alpha=0.2, lw=0)
                ax.semilogy(L, mean, color=B.COL[kind], ls=ls, lw=lw * 0.8, label=f'{B.LAB[kind]}, {src}')
        ax.set_xlabel(r'track length $\ell$ (days)'); ax.set_title(NAME[c])
        ax.set_ylim(1e-4, 1.2); ax.set_xlim(1, 28)
        panel_label(ax, pl, x=-0.05)
    axs[0].set_ylabel(r'share of tracks lasting $\geq\ell$ days')
    axs[1].legend(frameon=False, loc='upper right', fontsize=6.3)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(os.path.join(FIG, 'fig_h_tracks_null.pdf'))
    plt.close(fig)


def main():
    G = aggregates()
    fig_tp_daily(G)
    fig_tp_monthly(G)
    fig_traj_monthly()
    fig_tracks_null()
    P, valid, shape = projection_operator()
    anom, _, _ = build_anomalies()
    info = {'tracking_example': fig_tracking_example(anom, P, valid, shape),
            'event_2012': fig_event2012(anom, P, valid, shape)}
    json.dump(info, open(os.path.join(FIG, 'figure_info_hybrid.json'), 'w'), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == '__main__':
    main()
