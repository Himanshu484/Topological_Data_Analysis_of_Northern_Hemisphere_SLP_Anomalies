"""
C1 decomposition: Gaussian (ft) surrogates with the 1948-78 standardised
space-time structure (spectra and cross-spectra of z) but the 1979-2023
seasonal SD sigma(d, x) and mean state m(d, x).

  hybrid ~ ft_1979  ->  the 1979 step is an amplitude (variance) change
  hybrid ~ ft_1948  ->  it is a change in spatial/temporal structure

    python run_hybrid.py --reals 0 1 2
"""
import argparse
import os

import numpy as np

from common import DATA, START_YEAR, build_anomalies, field_features
from surrogates import Period


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reals', nargs='*', type=int, default=[0, 1, 2])
    a = ap.parse_args()
    anom, _, _ = build_anomalies()
    P79 = Period(anom[(1979 - START_YEAR) * 365:].copy())
    sig79, m79 = P79.sigma.copy(), P79.m.copy()
    del P79
    P = Period(anom[:(1979 - START_YEAR) * 365].copy())
    del anom
    # re-express a' of 1948-78 on the 1979-2023 amplitude: z is kept, sigma and m are swapped
    doy = P.doy
    P.ap *= (sig79 / P.sigma)[doy]
    P.sigma = sig79
    P.m = m79
    for r in a.reals:
        S = P.ft(np.random.default_rng(1000 * 1948 + r))
        field_features(S, 0, os.path.join(DATA, f'feat_hyb_1948_1978_r{r}_' + '{tag}.npz'))
        print('done', r, flush=True)


if __name__ == '__main__':
    main()
