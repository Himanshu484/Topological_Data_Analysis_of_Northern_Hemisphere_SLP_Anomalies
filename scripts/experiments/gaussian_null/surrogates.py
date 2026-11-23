"""
Multivariate surrogates of the NH SLP anomaly field (T x 37 x 144), per period.

Decomposition (per period P, e.g. 1979-2023):
    a(t, x)    anomaly w.r.t. the 76-year raw calendar-day climatology (pipeline)
    m(d, x)    mean of a over the years of P for calendar day d   (P's mean-state shift)
    a'         a - m
    sigma(d,x) sqrt of the across-year variance of a' on day d, smoothed to mean + 3 harmonics
    z          a' / sigma

Seasonal windows (the default, `windows=True`)
    Four overlapping windows centred on doy 15, 106, 197, 288 (mid-Jan, -Apr,
    -Jul, -Oct), each 121 days long, with weights w_s(d) whose squares ramp
    linearly over 30 days and sum to 1 on every calendar day.  For each window
    the days it covers are concatenated in time order (one 121-day block per
    year), standardised (z), and surrogated as below with its own random phases.
    The surrogate is  a_s = m + sigma * sum_s w_s z_s.  Each window keeps its
    own season's full space-time spectrum, the crossfade keeps the field
    continuous in time, and sum w_s^2 = 1 keeps the variance.  This keeps the
    seasonal cycle of spatial scale and of the temporal spectrum, which a single
    whole-record standardisation does not (`windows=False`, kept as a check):
    that version makes winter fields too small-scale, e.g. corr between two
    points 700 km apart 0.72 vs 0.77.
    Price: the 120 year-to-year seams per window make the within-window series
    lose correlation across a seam (<= tau/121 of lag-tau pairs, <1 % at lag 1).

(a) 'ft'   Prichard & Theiler (1994): Z = rfft_t(z); the SAME random phase
           phi(k) at every grid point, so every auto- and cross-spectrum of z in
           the window is kept exactly.  Gaussian.
(b) 'iaaft' multivariate IAAFT (Schreiber & Schmitz 2000) in each window,
           started from (a): spectral step Z(k,x) exp(i alpha(k)) with one common
           rotation alpha(k) = arg sum_x U(k,x) conj Z(k,x) (keeps all cross-
           spectra); amplitude step rank-remaps every grid point onto the real z
           values of that window (~121 x ny pooled values), and the iteration
           ends on an amplitude step, so each window's standardised marginal
           (skewness, tails) is exact and its spectrum approximate (error
           reported).  In the 30-day crossfades two skewed fields are summed,
           which dilutes skewness by at most w1^3 + w2^3 >= 0.71.
           A final rank-remap per grid point and calendar month (exact monthly
           marginals) was tried and rejected: with ~1400 autocorrelated values
           per month, neighbouring grid points get noisy, slightly different
           transforms, which adds pixel-scale roughness (all extrema per day
           88.9 -> 97.0; a remap pooled over all days adds only 1.5).
Both add back m, so each period's mean state is kept exactly.
"""
import numpy as np
import scipy.fft as sfft

MONTH_LEN = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
MONTH_OF_DOY = np.repeat(np.arange(12), MONTH_LEN)
CENTRES = [15, 106, 197, 288]
HALF, RAMP = 45.5, 30.0


def harmonic_smooth(x, nh=3):
    X = np.fft.rfft(x, axis=0)
    X[nh + 1:] = 0
    return np.fft.irfft(X, n=x.shape[0], axis=0)


def window_weights():
    """(4, 365) weights w_s(d) with sum_s w_s^2 = 1."""
    d = np.arange(365)
    W2 = []
    for c in CENTRES:
        dd = np.abs((d - c + 182) % 365 - 182)
        W2.append(np.clip((HALF + RAMP / 2 - dd) / RAMP, 0, 1))
    W2 = np.array(W2)
    W2 /= W2.sum(0, keepdims=True)
    return np.sqrt(W2)


class Period:
    def __init__(self, a, windows=True, chunk=888):
        """a: (ny*365, 37, 144) float32 slice of the pipeline anomaly for this period."""
        self.T = a.shape[0]
        self.ny = self.T // 365
        self.shape = a.shape[1:]
        self.X = int(np.prod(self.shape))
        self.chunk = chunk
        A = a.reshape(self.ny, 365, self.X)
        self.m = A.mean(axis=0, dtype=np.float64).astype(np.float32)
        self.ap = (A - self.m[None]).astype(np.float32).reshape(self.T, self.X)       # a'
        var = (self.ap.reshape(self.ny, 365, self.X).astype(np.float64) ** 2).mean(axis=0)
        self.sigma = np.sqrt(np.clip(harmonic_smooth(var), 1e-6, None)).astype(np.float32)
        self.doy = np.tile(np.arange(365), self.ny)
        self.midx = [np.where(MONTH_OF_DOY[self.doy] == k)[0] for k in range(12)]
        self.target = [np.sort(self.ap[ix], axis=0) for ix in self.midx]
        if windows:
            self.w = window_weights()
            self.widx = [np.where(self.w[s][self.doy] > 0)[0] for s in range(len(CENTRES))]
        else:
            self.w = np.ones((1, 365), np.float32)
            self.widx = [np.arange(self.T)]

    # ---- helpers -------------------------------------------------------
    def _z(self, idx):
        return self.ap[idx] / self.sigma[self.doy[idx]]

    def _rfft(self, z):
        Z = np.empty((z.shape[0] // 2 + 1, self.X), np.complex64)
        for c0 in range(0, self.X, self.chunk):
            Z[:, c0:c0 + self.chunk] = sfft.rfft(z[:, c0:c0 + self.chunk], axis=0)
        return Z

    def _irfft_rot(self, Z, rot, n):
        z = np.empty((n, self.X), np.float32)
        for c0 in range(0, self.X, self.chunk):
            z[:, c0:c0 + self.chunk] = sfft.irfft(Z[:, c0:c0 + self.chunk] * rot, n=n, axis=0)
        return z

    @staticmethod
    def _rank_remap(v, target_sorted):
        r = np.argsort(np.argsort(v, axis=0, kind='stable'), axis=0, kind='stable')
        return np.take_along_axis(target_sorted, r, axis=0)

    def _month_remap(self, s):
        for ix, tg in zip(self.midx, self.target):
            s[ix] = self._rank_remap(s[ix], tg)
        return s

    def _spec_err(self, zs, Z):
        """relative L2 error of |rfft(zs)| vs |Z| (one window, standardised)"""
        num = den = 0.0
        for c0 in range(0, self.X, self.chunk):
            U = np.abs(sfft.rfft(zs[:, c0:c0 + self.chunk], axis=0))[1:].astype(np.float64)
            R = np.abs(Z[1:, c0:c0 + self.chunk]).astype(np.float64)
            num += float(((U - R) ** 2).sum())
            den += float((R ** 2).sum())
        return float(np.sqrt(num / den))

    # ---- surrogates ----------------------------------------------------
    def _window_ft(self, idx, rng, iaaft_iter=0, errs=None):
        z = self._z(idx)
        n = len(idx)
        Z = self._rfft(z)
        phi = rng.uniform(0, 2 * np.pi, Z.shape[0])
        phi[0] = 0.0
        zs = self._irfft_rot(Z, np.exp(1j * phi).astype(np.complex64)[:, None], n)
        if iaaft_iter:
            zt = np.sort(z, axis=0)
            del z
            zs = self._rank_remap(zs, zt)
            if errs is not None:
                errs.append([self._spec_err(zs, Z)])
            for _ in range(iaaft_iter):
                acc = np.zeros(Z.shape[0], np.complex128)
                for c0 in range(0, self.X, self.chunk):
                    U = sfft.rfft(zs[:, c0:c0 + self.chunk], axis=0)
                    acc += (U * np.conj(Z[:, c0:c0 + self.chunk])).sum(axis=1)
                alpha = np.angle(acc)
                alpha[0] = 0.0
                zs = self._irfft_rot(Z, np.exp(1j * alpha).astype(np.complex64)[:, None], n)
                zs = self._rank_remap(zs, zt)
            if errs is not None:
                errs[-1].append(self._spec_err(zs, Z))
        return zs

    def _combine(self, rng, iaaft_iter, errs=None):
        s = np.zeros((self.T, self.X), np.float32)
        for k, idx in enumerate(self.widx):
            zs = self._window_ft(idx, rng, iaaft_iter, errs)
            d = self.doy[idx]
            s[idx] += (self.w[k][d][:, None] * self.sigma[d]) * zs
            del zs
        return s

    def _finish(self, s):
        s = s.reshape(self.ny, 365, self.X)
        s += self.m[None]
        return s.reshape(self.T, *self.shape)

    def ft(self, rng):
        """Gaussian surrogate a (T, 37, 144)."""
        return self._finish(self._combine(rng, 0))

    def iaaft(self, rng, n_iter=10, log=None, month_remap=False):
        errs = []
        s = self._combine(rng, n_iter, errs)
        if month_remap:        # exact monthly marginals, but adds pixel-scale noise (see README): off
            s = self._month_remap(s)
        if log is not None:
            log('    iaaft window spectral error (start -> end): '
                + ', '.join(f'{a:.3f}->{b:.3f}' for a, b in errs))
        return self._finish(s), errs
