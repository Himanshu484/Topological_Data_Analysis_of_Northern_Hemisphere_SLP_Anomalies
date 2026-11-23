"""
Two checks that the 6-hourly code is the daily code with a different step.

    python run_checks.py

validate_projection   the disc operator of grid.py, built for the 2.5 degree grid
                      (c = 86), against the operator of the daily pipeline
                      (gaussian_null/common.py) on the pixels the two images share
validate_surrogates   the chunked, memory-mapped surrogates of surrogates6h.py,
                      run with one slot per day, against gaussian_null/surrogates.py
                      on a synthetic skewed record with the same seed

Both need a few seconds and about 1 GB; nothing is downloaded.
"""
import os
import tempfile

import numpy as np

import grid as G


def validate_projection():
    import common                                             # gaussian_null/common.py
    G.GRIDS['2p5'] = dict(dlat=2.5, nlat=37, nlon=144, c=86)
    P_new, valid_new, img = G.projection_operator('2p5')
    P_old, valid_old, shape_old = common.projection_operator()
    r, c = np.unravel_index(valid_old, shape_old)             # old image is 172 wide, new one 173, same origin
    pos = np.searchsorted(valid_new, r * img[1] + c)
    assert (valid_new[pos] == r * img[1] + c).all(), 'an old pixel is missing from the new disc'
    f = np.random.default_rng(0).normal(size=(5, P_old.shape[1]))
    diff = max(np.abs(P_old @ x - (P_new @ x)[pos]).max() for x in f)
    print(f'projection: {len(valid_old)} shared pixels, largest difference {diff:.1e}')
    assert diff < 1e-9


def validate_surrogates(ny=6, shape=(8, 25)):
    import surrogates as S0                                   # gaussian_null/surrogates.py
    rng = np.random.default_rng(1)
    X = shape[0] * shape[1]
    T = ny * 365
    e = rng.normal(size=(T + 50, X))
    a = np.zeros((T + 50, X))
    for t in range(1, T + 50):                                # AR(1) in time, mixed in space, then skewed
        a[t] = 0.8 * a[t - 1] + e[t]
    a = a[50:] @ rng.normal(size=(X, X)) / np.sqrt(X)
    a = (np.exp(0.3 * a / a.std()) * (1 + 0.5 * np.sin(2 * np.pi * np.arange(T) / 365))[:, None]).astype(np.float32)
    a -= a.reshape(ny, 365, X).mean(0)[None].repeat(ny, 0).reshape(T, X)

    keep = (G.SPD, G.SPY, G.NY, G.T, G.Y0)
    G.SPD, G.SPY, G.NY, G.T = 1, 365, ny, T                   # one slot per day
    import surrogates6h as S6
    with tempfile.TemporaryDirectory() as tmp:
        np.save(os.path.join(tmp, 'a.npy'), a)
        P6 = S6.Period6h(os.path.join(tmp, 'a.npy'))
        P0 = S0.Period(a.reshape(T, *shape))
        for kind in ('ft', 'iaaft'):
            out = os.path.join(tmp, kind + '.npy')
            P6.surrogate(kind, 0, out)
            new = np.load(out)
            r0 = np.random.default_rng(1000 * G.Y0)
            old = (P0.ft(r0) if kind == 'ft' else P0.iaaft(r0)[0]).reshape(T, X)
            print(f'surrogate {kind}: largest difference {np.abs(new - old).max():.1e} (field SD {old.std():.2f}), '
                  f'correlation {np.corrcoef(new.ravel(), old.ravel())[0, 1]:.7f}')
            assert np.corrcoef(new.ravel(), old.ravel())[0, 1] > 0.9999
    G.SPD, G.SPY, G.NY, G.T, G.Y0 = keep


if __name__ == '__main__':
    validate_projection()
    validate_surrogates()
