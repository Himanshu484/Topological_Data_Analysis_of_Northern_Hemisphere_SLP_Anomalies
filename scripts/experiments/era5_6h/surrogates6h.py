"""
Gaussian ('ft') and amplitude-adjusted ('iaaft') surrogates of the 6-hourly,
1.5 degree ERA5 anomaly, and their persistence features.

The construction is that of scripts/experiments/gaussian_null/surrogates.py
(see its docstring for the reasoning), with one slot per 6 hours:

    m(s, x)      mean over years for each of the 1460 slots of the year
    sigma(s, x)  across-year standard deviation, smoothed to 3 annual harmonics
    z            (a - m) / sigma
    four 121-day windows (484 slots per year), cross-faded over 30 days,
    phase-randomised in time with the same phases at every grid point (ft),
    then iterated towards the observed distribution at every grid point with
    one common phase rotation per frequency (iaaft, 10 iterations).

What differs is only the implementation, because one window is now
21,296 x 14,640 values: the anomaly and the surrogate are memory-mapped files
in ERA6H_CACHE, rank remaps and FFTs run in column chunks on a thread pool,
and the rank remap scatters the sorted target values through one argsort
instead of ranking twice.  Peak memory is about 4 GB.

    python surrogates6h.py --reals 0 1 2 3 4        # ERA5, 1.5 degrees: ~20 min per realisation pair on 6 cores
    python surrogates6h.py --res r1 --reals 0 1 2 3 4   # NCEP-NCAR R1, 2.5 degrees: ~5 min per pair

For each realisation and kind it writes feat_<kind>_r<r>_<res>_{C,A}.npz and
diag_<kind>_r<r>_<res>.json to data/processed_data/experiments/era5_6h/, and
deletes the surrogate field.  Seeds: 1000 * 1979 + r, shared by ft and iaaft.
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.fft as sfft

import grid as G

CENTRES = [15, 106, 197, 288]
HALF, RAMP = 45.5, 30.0
CHUNK = 610
THREADS = int(os.environ.get('ERA6H_THREADS', 6))


def harmonic_smooth(x, nh=3):
    X = np.fft.rfft(x, axis=0)
    X[nh + 1:] = 0
    return np.fft.irfft(X, n=x.shape[0], axis=0)


def window_weights():
    d = np.arange(365)
    W2 = []
    for c in CENTRES:
        dd = np.abs((d - c + 182) % 365 - 182)
        W2.append(np.clip((HALF + RAMP / 2 - dd) / RAMP, 0, 1))
    W2 = np.array(W2)
    W2 /= W2.sum(0, keepdims=True)
    return np.sqrt(W2)


def chunks(X):
    return [(c0, min(c0 + CHUNK, X)) for c0 in range(0, X, CHUNK)]


class Period6h:
    def __init__(self, anom_path):
        self.A = np.load(anom_path, mmap_mode='r')                     # (T, X)
        self.T, self.X = self.A.shape
        s1 = np.zeros((G.SPY, self.X))
        s2 = np.zeros((G.SPY, self.X))
        for y in range(G.NY):
            a = np.asarray(self.A[y * G.SPY:(y + 1) * G.SPY], np.float64)
            s1 += a
            s2 += a * a
        self.m = (s1 / G.NY).astype(np.float32)
        var = s2 / G.NY - (s1 / G.NY) ** 2
        self.sigma = np.sqrt(np.clip(harmonic_smooth(var), 1e-6, None)).astype(np.float32)
        self.soy = np.tile(np.arange(G.SPY), G.NY)                      # slot of year
        self.doy = self.soy // G.SPD
        self.w = window_weights()
        self.widx = [np.where(self.w[s][self.doy] > 0)[0] for s in range(len(CENTRES))]
        self.pool = ThreadPoolExecutor(THREADS)

    def _z(self, idx):
        z = np.empty((len(idx), self.X), np.float32)
        step = 4840
        for a in range(0, len(idx), step):
            ii = idx[a:a + step]
            z[a:a + len(ii)] = (self.A[ii] - self.m[self.soy[ii]]) / self.sigma[self.soy[ii]]
        return z

    def _window(self, idx, rng, iaaft_iter, errs):
        z = self._z(idx)
        n = len(idx)
        nf = n // 2 + 1
        phi = rng.uniform(0, 2 * np.pi, nf)
        phi[0] = 0.0
        rot = np.exp(1j * phi).astype(np.complex64)[:, None]
        zs = np.empty_like(z)
        cs = chunks(self.X)
        if not iaaft_iter:
            def ft(c):
                zs[:, c[0]:c[1]] = sfft.irfft(sfft.rfft(z[:, c[0]:c[1]], axis=0) * rot, n=n, axis=0)
            list(self.pool.map(ft, cs))
            return zs
        Z = np.empty((nf, self.X), np.complex64)

        def first(c):
            Z[:, c[0]:c[1]] = sfft.rfft(z[:, c[0]:c[1]], axis=0)
            zs[:, c[0]:c[1]] = sfft.irfft(Z[:, c[0]:c[1]] * rot, n=n, axis=0)
            z[:, c[0]:c[1]].sort(axis=0)                                # z becomes the sorted target
        list(self.pool.map(first, cs))
        zt = z

        def remap(c):
            v = zs[:, c[0]:c[1]]
            order = np.argsort(v, axis=0, kind='stable')
            np.put_along_axis(v, order, zt[:, c[0]:c[1]], axis=0)

        def spec_err():
            def one(c):
                U = np.abs(sfft.rfft(zs[:, c[0]:c[1]], axis=0))[1:].astype(np.float64)
                R = np.abs(Z[1:, c[0]:c[1]]).astype(np.float64)
                return ((U - R) ** 2).sum(), (R ** 2).sum()
            r = np.array(list(self.pool.map(one, cs)))
            return float(np.sqrt(r[:, 0].sum() / r[:, 1].sum()))

        list(self.pool.map(remap, cs))
        e0 = spec_err()
        for _ in range(iaaft_iter):
            def corr(c):
                U = sfft.rfft(zs[:, c[0]:c[1]], axis=0)
                return (U * np.conj(Z[:, c[0]:c[1]])).sum(axis=1, dtype=np.complex128)
            alpha = np.angle(np.sum(list(self.pool.map(corr, cs)), axis=0))
            alpha[0] = 0.0
            ra = np.exp(1j * alpha).astype(np.complex64)[:, None]

            def step(c):
                zs[:, c[0]:c[1]] = sfft.irfft(Z[:, c[0]:c[1]] * ra, n=n, axis=0)
                remap(c)
            list(self.pool.map(step, cs))
        errs.append([e0, spec_err()])
        return zs

    def surrogate(self, kind, r, out_path):
        """Write the surrogate anomaly (T, X) float32 to out_path; returns the IAAFT spectral errors."""
        rng = np.random.default_rng(1000 * G.Y0 + r)
        s = np.lib.format.open_memmap(out_path, mode='w+', dtype=np.float32, shape=(self.T, self.X))
        errs = []
        for k, idx in enumerate(self.widx):
            zs = self._window(idx, rng, 10 if kind == 'iaaft' else 0, errs)
            soy = self.soy[idx]
            step = 4840
            for a in range(0, len(idx), step):                          # idx is sorted: plain += on fancy rows is safe
                ii = idx[a:a + step]
                s[ii] = s[ii] + (self.w[k][self.doy[ii]][:, None] * self.sigma[soy[a:a + step]]) * zs[a:a + len(ii)]
            del zs
        for y in range(G.NY):
            s[y * G.SPY:(y + 1) * G.SPY] += self.m
        s.flush()
        del s
        return errs


def surrogate_whole(P, r, out_path):
    """Gaussian surrogate from the whole record at once ('ftw'): no seasonal windows and no yearly seams.

    The standardised anomaly is phase-randomised over all T slots, so the lagged covariances are those of
    the record exactly (up to the single wrap-around), at the price of one spatial structure for all seasons.
    A control for the effect of the seams on the six-hour statistics.
    """
    rng = np.random.default_rng(1000 * G.Y0 + r)
    nf = P.T // 2 + 1
    phi = rng.uniform(0, 2 * np.pi, nf)
    phi[0] = 0.0
    rot = np.exp(1j * phi).astype(np.complex64)[:, None]
    A = P.A
    s = np.lib.format.open_memmap(out_path, mode='w+', dtype=np.float32, shape=(P.T, P.X))

    def one(c):
        m, sg = P.m[P.soy, c[0]:c[1]], P.sigma[P.soy, c[0]:c[1]]
        z = (A[:, c[0]:c[1]] - m) / sg
        s[:, c[0]:c[1]] = sfft.irfft(sfft.rfft(z, axis=0) * rot, n=P.T, axis=0) * sg + m
    list(ThreadPoolExecutor(2).map(one, [(a, min(a + 244, P.X)) for a in range(0, P.X, 244)]))
    s.flush()
    del s
    return []


def diagnostics(path, res='1p5'):
    """Area-weighted (20-90 N) skewness, winter and summer SD, and 6 h and 24 h autocorrelation."""
    A = np.load(path, mmap_mode='r')
    T, X = A.shape
    g = G.GRIDS[res]
    LAT = 90 - g['dlat'] * np.arange(g['nlat'])
    w = (np.cos(np.radians(LAT))[:, None] * np.ones((1, g['nlon']))).ravel()
    nh = (np.repeat(LAT, g['nlon']) >= 20)
    month = G.MONTH_OF_DOY[(np.arange(T) % G.SPY) // G.SPD]
    djf, jja = np.isin(month, [11, 0, 1]), np.isin(month, [5, 6, 7])
    out = {k: 0.0 for k in ('skew', 'neg', 'djf', 'jja', 'acf6h', 'acf24h')}
    wsum = w[nh].sum()
    for c0, c1 in chunks(X):
        k = nh[c0:c1]
        if not k.any():
            continue
        x = np.asarray(A[:, c0:c1][:, k], np.float64)
        ww = w[c0:c1][k] / wsum
        d = x - x.mean(0)
        sd = np.sqrt((d ** 2).mean(0))
        sk = (d ** 3).mean(0) / sd ** 3
        out['skew'] += float((ww * sk).sum())
        out['neg'] += float((ww * (sk < 0)).sum())
        out['djf'] += float((ww * (d[djf] ** 2).mean(0)).sum())
        out['jja'] += float((ww * (d[jja] ** 2).mean(0)).sum())
        out['acf6h'] += float((ww * (d[1:] * d[:-1]).mean(0) / sd ** 2).sum())
        out['acf24h'] += float((ww * (d[4:] * d[:-4]).mean(0) / sd ** 2).sum())
    out['djf'] = float(np.sqrt(out['djf']) / 100)
    out['jja'] = float(np.sqrt(out['jja']) / 100)
    return out


def main():
    import fields
    ap = argparse.ArgumentParser()
    ap.add_argument('--reals', nargs='*', type=int, default=[0, 1, 2])
    ap.add_argument('--kinds', nargs='*', default=['ft', 'iaaft'])
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--res', default='1p5', help="'1p5' (ERA5) or 'r1' (NCEP-NCAR R1, 2.5 degrees)")
    a = ap.parse_args()
    res = a.res
    anom = os.path.join(G.CACHE, f'anom_{res}.npy')
    dpath = os.path.join(G.DATA, f'diag_real_{res}.json')
    if not os.path.exists(dpath):
        json.dump(diagnostics(anom, res), open(dpath, 'w'), indent=1)
    P = None
    for r in a.reals:
        for kind in a.kinds:
            name = f'{kind}_r{r}'
            dj = os.path.join(G.DATA, f'diag_{name}_{res}.json')
            if os.path.exists(dj):
                print(f'{name} exists, skipped', flush=True)
                continue
            if P is None:
                t0 = time.time()
                P = Period6h(anom)
                print(f'period prepared in {time.time() - t0:.0f}s', flush=True)
            t0 = time.time()
            path = os.path.join(G.CACHE, f'sur_{name}_{res}.npy')
            errs = surrogate_whole(P, r, path) if kind == 'ftw' else P.surrogate(kind, r, path)
            t1 = time.time()
            info = diagnostics(path, res)
            info['spectral_error'] = errs
            fields.features_from_array(path, res, name, workers=a.workers)
            os.remove(path)
            json.dump(info, open(dj, 'w'), indent=1)
            print(f'{name}: surrogate {t1 - t0:.0f}s, diagnostics and features {time.time() - t1:.0f}s; '
                  f'skew {info["skew"]:+.3f} sd DJF {info["djf"]:.2f} JJA {info["jja"]:.2f} acf6h {info["acf6h"]:.3f} '
                  f'acf24h {info["acf24h"]:.3f}', flush=True)


if __name__ == '__main__':
    main()
