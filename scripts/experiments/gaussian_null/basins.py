"""
Merge-tree basins of persistence features on a polar disc.

For a disc g (172 x 172, padding = g[0,0]) the superlevel union-find gives every
H0 pair = every 1-cycle of the manuscript (Alexander duality, Section 2.5):

    peak    the extremum pixel                        (= death cell)
    saddle  the pixel at which it merges              (= birth cell)
    parent  the peak it merges into, -1 = disc rim
    basin   km^2 of the 8-connected component of {g > saddle value} containing
            the peak (the region inside the outermost closed contour enclosing
            no higher peak); computed at the moment of death, so it is exact
    pole    True if that basin contains the pole pixel

`level_area` gives the km^2 of the component of {g > level} containing a peak,
for arbitrary per-feature levels (e.g. half depth: peak - depth/2).

Apply to `slp_sup` discs for cyclones (g = -f) and `slp_sub` for anticyclones.
Identical pairs to scripts/prototype/pers_features.py (validated in
validate_basins.py).

Pixel area: the disc is azimuthal-equidistant with the pole at (86, 86) and the
equator at r = 86 px, so a 1 px^2 cell at radius r px covers
    R^2 s^2 sin(s r)/(s r),   s = (pi/2)/86 rad per px
(13,550 km^2 at the pole, 8,630 km^2 at the equator).
"""
import ctypes
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_LIB = os.path.join(HERE, 'libufbasin.so')
RE_KM = 6371.0088
C_PIX = 86
H = W = 172
POLE = C_PIX * W + C_PIX


def _load():
    src = os.path.join(HERE, 'uf_basin.c')
    if not os.path.exists(_LIB) or os.path.getmtime(_LIB) < os.path.getmtime(src):
        subprocess.check_call(['cc', '-O3', '-shared', '-fPIC', '-o', _LIB, src])
    lib = ctypes.CDLL(_LIB)
    P = np.ctypeslib.ndpointer
    d, u8, i4 = P(np.float64, flags='C'), P(np.uint8, flags='C'), P(np.int32, flags='C')
    lib.uf_features.restype = ctypes.c_int
    lib.uf_features.argtypes = [d, u8, d, i4, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                i4, i4, i4, d, i4, ctypes.c_int]
    lib.uf_level_area.restype = None
    lib.uf_level_area.argtypes = [d, u8, d, i4, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                  i4, d, i4, ctypes.c_int, d]
    return lib


_lib = _load()


def pixel_area_km2(h=H, w=W, c=C_PIX):
    rr, cc = np.mgrid[:h, :w]
    r = np.hypot(rr - c, cc - c)
    s = (np.pi / 2) / c
    x = s * r
    f = np.where(r > 0, np.sin(x) / np.where(x > 0, x, 1), 1.0)
    return (RE_KM * s) ** 2 * f


PAREA = pixel_area_km2()


class Disc:
    """One disc prepared for sweeps (sort order computed once)."""

    def __init__(self, g, sea=None):
        g = np.ascontiguousarray(g, dtype=np.float64)
        self.h, self.w = g.shape
        self.val = g.ravel()
        if sea is None:
            sea = g == g[0, 0]
        self.sea = np.ascontiguousarray(sea.ravel(), dtype=np.uint8)
        self.parea = np.ascontiguousarray(PAREA.ravel() if g.shape == PAREA.shape
                                          else pixel_area_km2(*g.shape, c=g.shape[0] // 2).ravel())
        o = np.argsort(-self.val, kind='stable')        # == lexsort((arange, -val))
        self.order = np.ascontiguousarray(o[~self.sea.astype(bool)[o]], dtype=np.int32)

    def features(self):
        """dict of arrays: peak, saddle (values), peak_idx, saddle_idx, parent_idx, basin_km2, pole."""
        m = len(self.order)
        o = [np.empty(m, np.int32) for _ in range(3)]
        oa = np.empty(m, np.float64)
        op = np.empty(m, np.int32)
        n = _lib.uf_features(self.val, self.sea, self.parea, self.order, m, self.h, self.w,
                             POLE if (self.h, self.w) == (H, W) else -1, o[0], o[1], o[2], oa, op, m)
        pk, sd, par = (a[:n] for a in o)
        return dict(peak=self.val[pk], saddle=self.val[sd], peak_idx=pk, saddle_idx=sd,
                    parent_idx=par, basin_km2=oa[:n], pole=op[:n].astype(bool))

    def level_area(self, peak_idx, level):
        """km^2 of the component of {g > level} containing each peak (-1 if not defined)."""
        qp = np.ascontiguousarray(peak_idx, dtype=np.int32)
        ql = np.ascontiguousarray(level, dtype=np.float64)
        qo = np.ascontiguousarray(np.argsort(-ql, kind='stable'), dtype=np.int32)
        out = np.empty(len(qp), np.float64)
        _lib.uf_level_area(self.val, self.sea, self.parea, self.order, len(self.order), self.h, self.w,
                           qp, ql, qo, len(qp), out)
        return out


def features_with_basins(g, sea=None, half=True):
    """Convenience: features plus half-depth area (level = peak - depth/2)."""
    D = Disc(g, sea)
    F = D.features()
    if half:
        F['half_km2'] = D.level_area(F['peak_idx'], F['peak'] - 0.5 * (F['peak'] - F['saddle']))
    return F
