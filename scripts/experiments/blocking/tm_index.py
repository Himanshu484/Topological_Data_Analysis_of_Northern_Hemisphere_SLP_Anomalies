"""
Reference blocking index: 2-D Tibaldi-Molteni on Z500 (Scherrer et al. 2006,
with the Davini et al. 2012 additions), 1948-2023, 2.5 deg NCEP-NCAR R1.

  instantaneous blocking (IB) at (phi0, lambda), phi0 = 30..75N:
      GHGS  = (Z(phi0) - Z(phi0-15)) / 15      > 0
      GHGN  = (Z(phi0+15) - Z(phi0)) / 15      < -10 gpm / deg lat
      GHGS2 = (Z(phi0-15) - Z(phi0-30)) / 15   < -5  gpm / deg lat  (removes subtropical lows)
  large-scale blocking (LSB): IB inside a run of >= 6 consecutive longitudes (15 deg)
      of IB at phi0 or +-2.5 deg latitude
  blocking event (BE) gridpoint-day: LSB within +-2.5 deg lat, +-5 deg lon of the
      point on >= 5 consecutive days
  events: space-time connected components of the BE mask (26-connectivity, periodic in
      longitude), phi0 >= 40N for evaluation (the 30-37.5N rows are kept in the maps)

Outputs (CACHE): tm_be.npy  bool (76*365, 19, 144)  rows phi0 = 75, 72.5, ..., 30
                 tm_events.npz  per-event table and per-event-day centroids
Figures (OUT):   tm_climatology.png, tm_lon_profile.png, tm_seasonal_cycle.png
Tables (OUT):    tm_climatology.txt

    python scripts/experiments/blocking/tm_index.py
"""
import os

import numpy as np
from scipy import ndimage

from common import (CACHE, OUT, LAT, LON, Y0, Y1, EVAL0, EVAL1, MONTH, SEASON, SEASONS, SECTORS,
                    in_sector, circmean_lon, slot_date, date_slot)

I75, I30 = 6, 24                      # NCEP rows of 75N and 30N
ROWS = np.arange(I75, I30 + 1)        # 19 rows phi0 = 75 .. 30
PHI0 = LAT[ROWS]
MIN_LAT_EVAL = 40.0


def runs_periodic(mask1d, minlen):
    """Boolean mask of elements lying in periodic runs of True of length >= minlen."""
    n = len(mask1d)
    if mask1d.all():
        return mask1d.copy()
    if not mask1d.any():
        return mask1d.copy()
    k = int(np.argmin(mask1d))           # a False element: roll so the series starts there
    m = np.roll(mask1d, -k)
    out = np.zeros(n, bool)
    lab, nl = ndimage.label(m)
    if nl:
        sizes = ndimage.sum(m, lab, np.arange(1, nl + 1))
        good = np.where(sizes >= minlen)[0] + 1
        out = np.isin(lab, good)
    return np.roll(out, k)


def runs_time(mask, minlen):
    """mask (T, ...) -> True where the element is inside a run of >= minlen consecutive True in time."""
    T = mask.shape[0]
    fwd = np.zeros(mask.shape, np.int32)
    bwd = np.zeros(mask.shape, np.int32)
    fwd[0] = mask[0]
    for t in range(1, T):
        fwd[t] = np.where(mask[t], fwd[t - 1] + 1, 0)
    bwd[-1] = mask[-1]
    for t in range(T - 2, -1, -1):
        bwd[t] = np.where(mask[t], bwd[t + 1] + 1, 0)
    return mask & (fwd + bwd - 1 >= minlen)


def compute_be():
    Z = np.load(os.path.join(CACHE, 'z500_full.npy'), mmap_mode='r')
    ny = Z.shape[0]
    T = ny * 365
    IB = np.zeros((T, len(ROWS), 144), bool)
    for j in range(ny):
        z = np.asarray(Z[j], np.float32)                       # (365, 37, 144)
        z0 = z[:, ROWS]
        zN = z[:, ROWS - 6]
        zS = z[:, ROWS + 6]
        zS2 = z[:, ROWS + 12]
        IB[j * 365:(j + 1) * 365] = ((z0 - zS) / 15 > 0) & ((zN - z0) / 15 < -10) & ((zS - zS2) / 15 < -5)
    tol = IB.copy()
    tol[:, 1:] |= IB[:, :-1]
    tol[:, :-1] |= IB[:, 1:]
    LSB = np.zeros_like(IB)
    for t in range(T):
        for r in range(len(ROWS)):
            if IB[t, r].any():
                LSB[t, r] = IB[t, r] & runs_periodic(tol[t, r], 6)
    box = LSB.copy()
    box[:, 1:] |= LSB[:, :-1]
    box[:, :-1] |= LSB[:, 1:]
    box2 = box.copy()
    for s in (1, 2):
        box2 |= np.roll(box, s, axis=2) | np.roll(box, -s, axis=2)
    BE = runs_time(box2, 5)
    return IB, LSB, BE


def label_periodic_3d(mask):
    lab, n = ndimage.label(mask, structure=np.ones((3, 3, 3), bool))
    # merge across the longitude seam (0 <-> 357.5), 26-connectivity
    parent = np.arange(n + 1)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    L, R = lab[..., 0], lab[..., -1]
    T, H = L.shape
    for dt in (-1, 0, 1):
        for dh in (-1, 0, 1):
            a = L[max(0, -dt):T - max(0, dt), max(0, -dh):H - max(0, dh)]
            b = R[max(0, dt):T + min(0, dt), max(0, dh):H + min(0, dh)]
            k = (a > 0) & (b > 0)
            for x, y in set(zip(a[k].tolist(), b[k].tolist())):
                rx, ry = find(x), find(y)
                if rx != ry:
                    parent[max(rx, ry)] = min(rx, ry)
    roots = np.array([find(i) for i in range(n + 1)])
    uniq, new = np.unique(roots, return_inverse=True)
    return new[lab], len(uniq) - 1


def events_table(BE):
    rows_eval = PHI0 >= MIN_LAT_EVAL
    M = BE.copy()
    M[:, ~rows_eval] = False
    lab, n = label_periodic_3d(M)
    objs = ndimage.find_objects(lab)
    ev = []
    day_rows = []
    lab2eid = -np.ones(n + 1, int)
    coslat = np.cos(np.radians(PHI0))
    for e in range(1, n + 1):
        sl = objs[e - 1]
        if sl is None:
            continue
        sub = lab[sl[0], :, :] == e
        t0 = sl[0].start
        days = []
        for k in range(sub.shape[0]):
            rr, cc = np.where(sub[k])
            if len(rr) == 0:
                continue
            w = coslat[rr]
            la = float((PHI0[rr] * w).sum() / w.sum())
            lo = float(circmean_lon(LON[cc], w))
            days.append((t0 + k, la, lo, float(w.sum())))
        if len(days) < 5:
            continue
        d = np.array(days)
        lo_ev = circmean_lon(d[:, 2], d[:, 3])
        la_ev = float((d[:, 1] * d[:, 3]).sum() / d[:, 3].sum())
        eid = len(ev)
        lab2eid[e] = eid
        ev.append((eid, int(d[0, 0]), int(d[-1, 0]), len(d), la_ev, lo_ev, float(d[:, 3].mean())))
        for x in d:
            day_rows.append((eid, int(x[0]), x[1], x[2], x[3]))
    ev = np.array(ev)
    dr = np.array(day_rows)
    t, r, c = np.nonzero(lab)
    pe = lab2eid[lab[t, r, c]]
    k = pe >= 0
    pts = np.stack([t[k], r[k], c[k], pe[k]], 1).astype(np.int32)
    return ev, dr, pts


def main():
    IB, LSB, BE = compute_be()
    np.save(os.path.join(CACHE, 'tm_be.npy'), BE)
    np.save(os.path.join(CACHE, 'tm_ib.npy'), IB)
    ev, dr, pts = events_table(BE)
    np.save(os.path.join(CACHE, 'tm_points.npy'), pts)   # (t, row in ROWS, col, eid)
    sect = np.array([next(s for s in SECTORS if in_sector(np.array([lo]), s)[0]) for lo in ev[:, 5]])
    np.savez(os.path.join(CACHE, 'tm_events.npz'), eid=ev[:, 0].astype(int), t0=ev[:, 1].astype(int),
             t1=ev[:, 2].astype(int), dur=ev[:, 3].astype(int), lat=ev[:, 4], lon=ev[:, 5],
             size=ev[:, 6], sector=sect,
             d_eid=dr[:, 0].astype(int), d_t=dr[:, 1].astype(int), d_lat=dr[:, 2], d_lon=dr[:, 3],
             d_size=dr[:, 4])

    # ---------------- climatology ----------------
    ny = Y1 - Y0 + 1
    tt = np.arange(ny * 365)
    yr = Y0 + tt // 365
    slot = tt % 365 + 1
    ev_mask = (yr >= EVAL0) & (yr <= EVAL1)
    lines = []
    lines.append(f'2-D TM (Davini 2012) blocking-event frequency, {EVAL0}-{EVAL1}, NCEP-NCAR R1 Z500\n')
    freq = {}
    for s in SEASONS + ['ANN']:
        m = ev_mask & ((SEASON[slot - 1] == s) if s != 'ANN' else True)
        freq[s] = 100 * BE[m].mean(0)
        f = freq[s]
        r, c = np.unravel_index(np.argmax(f), f.shape)
        lines.append(f'{s}: max BE frequency {f.max():.1f}% at {PHI0[r]:.1f}N {LON[c]:.1f}E')
        for sec in SECTORS:
            cm = in_sector(LON, sec)
            fr = f[(PHI0 >= 40)][:, cm]
            rr, cc = np.unravel_index(np.argmax(fr), fr.shape)
            lines.append(f'    {sec}: sector max {fr.max():.1f}% at {PHI0[PHI0 >= 40][rr]:.1f}N '
                         f'{LON[cm][cc]:.1f}E; sector-mean (40-75N) {fr.mean():.1f}%')
    # events per year, per sector and season
    sel = (Y0 + ev[:, 1].astype(int) // 365 >= EVAL0) & (Y0 + ev[:, 1].astype(int) // 365 <= EVAL1)
    nyr = EVAL1 - EVAL0 + 1
    lines.append(f'\nEvents (phi0 >= 40N, >= 5 d): {sel.sum()} in {nyr} yr = {sel.sum() / nyr:.1f}/yr, '
                 f'median duration {np.median(ev[sel, 3]):.0f} d, mean {ev[sel, 3].mean():.1f} d, '
                 f'90th pct {np.percentile(ev[sel, 3], 90):.0f} d')
    onset_season = SEASON[(ev[:, 1].astype(int) % 365)]
    hdr = 'sector   ' + '  '.join(f'{s:>5}' for s in SEASONS) + '   total'
    lines.append('events per year by sector (rows) and onset season (cols):')
    lines.append(hdr)
    for sec in SECTORS:
        cnt = [((sect == sec) & sel & (onset_season == s)).sum() / nyr for s in SEASONS]
        lines.append(f'{sec:8} ' + '  '.join(f'{c:5.2f}' for c in cnt) + f'   {sum(cnt):5.2f}')
    # case studies
    lines.append('\nCase checks (BE gridpoints, phi0 >= 40, in the stated box):')
    for (y, m0, d0, m1, d1, lo0, lo1, name) in [(2003, 8, 1, 8, 13, -20, 40, 'Aug 2003 Europe 20W-40E'),
                                                 (2012, 1, 15, 2, 20, 20, 90, 'Jan-Feb 2012 Eurasia 20-90E')]:
        s0, s1 = date_slot(m0, d0), date_slot(m1, d1)
        cm = np.mod(LON - lo0, 360) <= (lo1 - lo0) % 360
        for s in range(s0, s1 + 1):
            t = (y - Y0) * 365 + s - 1
            b = BE[t][PHI0 >= 40][:, cm]
            if b.any():
                rr, cc = np.where(b)
                la = PHI0[PHI0 >= 40][rr]
                lo = LON[cm][cc]
                lines.append(f'  {name}: {slot_date(y, s)}  {b.sum():3d} pts  lat {la.min():.1f}-{la.max():.1f}  '
                             f'lon {lo.min():.1f}-{lo.max():.1f}')
            else:
                lines.append(f'  {name}: {slot_date(y, s)}  --')
    txt = '\n'.join(lines)
    print(txt)
    open(os.path.join(OUT, 'tm_climatology.txt'), 'w').write(txt + '\n')

    # ---------------- figures ----------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    fig = plt.figure(figsize=(12, 10))
    for k, s in enumerate(SEASONS):
        ax = fig.add_subplot(2, 2, k + 1, projection=ccrs.NorthPolarStereo())
        ax.set_extent([-180, 180, 25, 90], ccrs.PlateCarree())
        f = np.concatenate([freq[s], freq[s][:, :1]], axis=1)
        lo = np.append(LON, 360)
        pc = ax.contourf(lo, PHI0, f, levels=np.arange(0, 22, 2), transform=ccrs.PlateCarree(),
                         cmap='YlOrRd', extend='max')
        ax.coastlines(lw=0.5)
        ax.set_title(f'{s}: BE frequency (% days), {EVAL0}-{EVAL1}')
        plt.colorbar(pc, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'tm_climatology.png'), dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    for s in SEASONS:
        ax.plot(LON, freq[s][PHI0 >= 40].max(0), label=s)
    ax.set_xlabel('longitude (deg E)')
    ax.set_ylabel('max over 40-75N of BE freq (%)')
    ax.legend()
    ax.set_title('Longitude profile of 2-D TM blocking frequency')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'tm_lon_profile.png'), dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for sec in SECTORS:
        cm = in_sector(LON, sec)
        mon = [100 * BE[ev_mask & (MONTH[slot - 1] == mm)][:, PHI0 >= 40][:, :, cm].any(axis=(1, 2)).mean()
               for mm in range(1, 13)]
        ax.plot(range(1, 13), mon, marker='o', label=sec)
    ax.set_xlabel('month')
    ax.set_ylabel('% of days with a BE gridpoint in sector')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'tm_seasonal_cycle.png'), dpi=110)
    plt.close(fig)


if __name__ == '__main__':
    main()
