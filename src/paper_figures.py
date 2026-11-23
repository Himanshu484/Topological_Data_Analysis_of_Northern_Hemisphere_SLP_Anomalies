"""
Generator of the pedagogical figures of the manuscript (Figs 1-4 and the
example-day filtration, Fig. 6).

Each figure file is written by exactly one ``fig*`` function below, straight
into the manuscript's ``figures/`` directory, with the file name the manuscript
expects.  ``scripts/make_paper_figures.py`` is the command-line entry point.

Conventions (these are the ones the analysis uses, not plotting choices):

* Fields are the processed discs in ``data/processed_data/SLP_data_years``: a
  172 x 172 azimuthal-equidistant disc (pole at the centre, equator on the rim,
  disc = pixels with radius <= 86).  The ``sub`` file holds the anomaly f in Pa,
  the ``sup`` file holds -f; both are padded outside the disc.
* Persistence diagrams, representative cycles and death simplices are the
  Ripserer.jl output in ``representative_data`` (V-construction).  Superlevel
  (``sup``) values are stored for -f and are sign-flipped back to f for display.
* Maps are drawn in the azimuthal-equidistant projection the analysis itself
  uses, by placing the disc pixels directly (no regridding), so every contour
  and death vertex sits exactly on the pixel it was computed on.
* Dates are mapped to file slots on the 365-day no-leap calendar
  (``disc_utils.date_to_noleap_slot``).
"""
import datetime as dt
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, 'src') not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, 'src'))

import disc_utils as ft          # noqa: E402
from tda_metrics import wasserstein    # noqa: E402

# figures go to $MANUSCRIPT_DIR/figures, else ../manuscript/figures next to the repository, else ./paper_outputs/figures
_MS = os.environ.get('MANUSCRIPT_DIR') or os.path.join(os.path.dirname(ROOT), 'manuscript')
FIG_DIR = os.path.join(_MS if os.path.isdir(_MS) else os.path.join(ROOT, 'paper_outputs'), 'figures')
PROC = os.path.join(ROOT, 'data', 'processed_data')
SLP_DIR = os.path.join(PROC, 'SLP_data_years')
REP_DIR = os.path.join(PROC, 'representative_data')

N = 172                    # disc image size (pixels)
C = N // 2                 # disc centre and radius (pixels)
R_EARTH = 6371000.0        # sphere used for the plotting projection (m)
PIX = R_EARTH * np.pi / 2 / C          # metres per pixel along a meridian
EXTENT = ((-0.5 - C) * PIX, (N - 0.5 - C) * PIX,
          (-0.5 - C) * PIX, (N - 0.5 - C) * PIX)
_YY, _XX = np.mgrid[:N, :N]
DISC = np.hypot(_XX - C, _YY - C) <= C

# colours used consistently across figures
ORANGE, GREEN = '#f39c12', '#2ecc71'
RED, BLUE, CYAN, YELLOW = '#d62728', '#1f4fd6', '#00c8ff', '#ffe600'
LABEL = {'sup': '1-cyclones (superlevel)', 'sub': '1-anticyclones (sublevel)'}

JPG = dict(dpi=300, bbox_inches='tight', pil_kwargs={'quality': 92})

# --------------------------------------------------------------------------- #
# Synthetic data (Sec. 2).  These arrays ARE the examples in the manuscript.   #
# --------------------------------------------------------------------------- #

_L = [[10] * 9,
      [10, 10, 25, 53, 63, 53, 25, 10, 10],
      [10, 25, 74, 112, 127, 112, 74, 25, 10],
      [10, 53, 112, 164, 191, 164, 112, 53, 10],
      [10, 63, 127, 191, 254, 191, 127, 63, 10]]
_R = [[220] * 9,
      [220, 220, 202, 182, 175, 182, 202, 220, 220],
      [220, 202, 167, 140, 130, 140, 167, 202, 220],
      [220, 182, 140, 103, 85, 103, 140, 182, 220],
      [220, 175, 130, 85, 1, 85, 130, 175, 220]]
#: the 9 x 18 grayscale image of Figs. 2-3 (values printed in Fig. 2b)
GRID = np.hstack([np.array(_L + _L[3::-1]), np.array(_R + _R[3::-1])]).astype(float)

#: thresholds shown in the Fig. 3 filtration strips, top to bottom
SYN_THRESHOLDS = {'sup': [253, 200, 100, 10, 0], 'sub': [255, 200, 100, 10, 2]}

#: Fig. 4 sequence: background, spot centres (row, col), and per-frame
#: (peak, sigma) for spot A (upper left) and spot B (lower right)
#: (spot tails must round to the background before reaching the image edge or
#: the other spot, or both loops would not be born on the background level)
SEQ_SIZE, SEQ_BG = 28, 30
SEQ_CENTRES = {'A': (8, 8), 'B': (19, 19)}
SEQ_FRAMES = [{'A': (200, 1.3), 'B': (150, 1.0)},
              {'A': (230, 1.7), 'B': (175, 1.3)},
              {'A': (245, 2.2), 'B': (120, 0.8)}]


def synthetic_sequence():
    """The three frames of Fig. 4 as integer arrays."""
    frames = []
    for spec in SEQ_FRAMES:
        f = np.full((SEQ_SIZE, SEQ_SIZE), float(SEQ_BG))
        for k, (peak, sig) in spec.items():
            r0, c0 = SEQ_CENTRES[k]
            d2 = (_mg(SEQ_SIZE)[0] - r0) ** 2 + (_mg(SEQ_SIZE)[1] - c0) ** 2
            f += (peak - SEQ_BG) * np.exp(-d2 / (2 * sig ** 2))
        frames.append(np.clip(np.round(f), 0, 255))
    return frames


def _mg(n):
    return np.mgrid[:n, :n]


def h1_diagram(img, superlevel=False, construction='T'):
    """H1 (birth, death) pairs of a small image, in image units.

    ``construction='T'`` treats pixels as top cells (Sec. 2.3); ``'V'`` treats
    them as vertices (the construction used for every result).
    """
    import cripser
    import tcripser
    mod = tcripser if construction == 'T' else cripser
    g = -img if superlevel else img
    D = mod.computePH(np.ascontiguousarray(g, dtype=float), maxdim=1)
    D = D[(D[:, 0] == 1) & np.isfinite(D[:, 2])]
    s = -1.0 if superlevel else 1.0
    out = np.column_stack([s * D[:, 1], s * D[:, 2], D[:, 6], D[:, 7]])
    return out[np.argsort(-np.abs(out[:, 1] - out[:, 0]))]


# --------------------------------------------------------------------------- #
# Data access                                                                  #
# --------------------------------------------------------------------------- #

def slot(date):
    s = ft.date_to_noleap_slot(date)
    if s is None:
        raise ValueError(f'{date} is 29 February, which has no file slot')
    return s


def load_disc(date, tag):
    """Stored disc for ``tag`` (Pa; ``sup`` holds -f), padding included."""
    return np.load(os.path.join(SLP_DIR, str(date.year),
                                f'slp_{tag}_{date.year}_day_{slot(date)}.npy')).astype(float)


def load_reps(date, tag):
    """Production diagram of one day.

    Returns a dict with ``birth``/``death`` in hPa *of the anomaly f* (so the
    superlevel diagram lies below the diagonal), ``depth`` (hPa, >= 0),
    ``reps`` (representative edge lists) and ``dsv`` (death-cube vertices).
    """
    fp = os.path.join(REP_DIR, str(date.year), f'slp_{tag}_{date.year}_day_{slot(date)}.json')
    J = json.load(open(fp))
    s = -1.0 if tag == 'sup' else 1.0
    b = s * np.asarray(J['births'], float) / 100.0
    d = s * np.asarray(J['deaths'], float) / 100.0
    return {'birth': b, 'death': d, 'depth': np.abs(d - b),
            'stored_death': np.asarray(J['deaths'], float),
            'reps': J['list_data'], 'dsv': J['death_simplex_vertices']}


def death_vertex(R, i, date, tag):
    """(row, col) of the critical vertex of feature ``i``'s death cube."""
    return ft.resolve_death_vertex(R['dsv'][i], R['stored_death'][i],
                                   disc=load_disc(date, tag))


# --------------------------------------------------------------------------- #
# Drawing helpers                                                              #
# --------------------------------------------------------------------------- #

def _plt(fs=9):
    """pyplot with font sizes for a figure drawn at its printed size."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': fs, 'axes.titlesize': fs, 'axes.labelsize': fs,
                         'xtick.labelsize': fs - 1, 'ytick.labelsize': fs - 1,
                         'legend.fontsize': fs - 1, 'figure.dpi': 100})
    return plt


def ae_crs():
    import cartopy.crs as ccrs
    globe = ccrs.Globe(ellipse=None, semimajor_axis=R_EARTH, semiminor_axis=R_EARTH)
    return ccrs.AzimuthalEquidistant(central_longitude=0.0, central_latitude=90.0, globe=globe)


def pix_to_xy(row, col):
    """Disc pixel -> projected (x, y) in the azimuthal-equidistant CRS."""
    return (np.asarray(col, float) - C) * PIX, (np.asarray(row, float) - C) * PIX


def check_projection(n=500, seed=0):
    """Assert that pixel placement agrees with the analysis' own inverse map.

    Every pixel's (lat, lon) from ``disc_utils._rowcol_to_latlon`` is
    projected with cartopy and compared with ``pix_to_xy``.  Returns the maximum
    discrepancy in pixels.
    """
    import cartopy.crs as ccrs
    rng = np.random.default_rng(seed)
    rc = np.argwhere(DISC)
    rc = rc[rng.choice(len(rc), size=min(n, len(rc)), replace=False)]
    ae, pc = ae_crs(), ccrs.PlateCarree()
    worst = 0.0
    for r, c in rc:
        lat, lon = ft._rowcol_to_latlon(r, c, N, (0, 90), (0, 360))
        x, y = ae.transform_point(lon, lat, pc)
        x0, y0 = pix_to_xy(r, c)
        worst = max(worst, np.hypot(x - x0, y - y0) / PIX)
    if worst > 0.01:
        raise AssertionError(f'map placement disagrees with the analysis by {worst:.3f} px')
    return worst


def map_axes(fig, spec):
    """A round azimuthal-equidistant map axis covering the disc."""
    import matplotlib.path as mpath
    ax = fig.add_subplot(spec, projection=ae_crs())
    lim = (C + 1.0) * PIX
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    th = np.linspace(0, 2 * np.pi, 361)
    ax.set_boundary(mpath.Path(np.c_[np.cos(th), np.sin(th)] * (C + 0.5) * PIX),
                    transform=ax.transData)
    return ax


def decorate_map(ax, lw=0.5):
    import cartopy.crs as ccrs
    ax.coastlines(resolution='110m', linewidth=lw, color='k')
    ax.gridlines(crs=ccrs.PlateCarree(), linewidth=0.3, color='0.45', alpha=0.7,
                 xlocs=np.arange(-180, 180, 60), ylocs=[0, 30, 60])


def draw_pd(ax, R, highlight=(), lim=None, fontsize=9, point_size=8, star_size=170):
    """Persistence diagram in hPa; ``highlight`` = [(index, colour, marker)]."""
    b, d = R['birth'], R['death']
    hi = {i for i, _, _ in highlight}
    rest = [k for k in range(len(b)) if k not in hi]
    if lim is None:
        lo = min(b.min(), d.min()); up = max(b.max(), d.max())
        pad = 0.06 * (up - lo)
        lim = (lo - pad, up + pad)
    ax.plot(lim, lim, color='k', lw=0.8, zorder=1)
    ax.axhline(0, color='0.6', lw=0.6, ls='--', zorder=0)
    ax.axvline(0, color='0.6', lw=0.6, ls='--', zorder=0)
    ax.scatter(b[rest], d[rest], s=point_size, c='k', zorder=2)
    # labels are stacked in the empty half of the diagram (below the diagonal
    # for a sublevel diagram, above it for a superlevel one) with leader lines,
    # so they cannot collide with each other or with the points
    upper_empty = np.mean(d < b) > 0.5
    for k, (i, col, mk) in enumerate(highlight):
        big = star_size if mk == '*' else 0.35 * star_size
        ax.scatter([b[i]], [d[i]], s=big, c=col, marker=mk, edgecolors='k',
                   linewidths=0.6, zorder=5)
        pos, ha, va = (((0.04, 0.96 - 0.2 * k), 'left', 'top') if upper_empty
                       else ((0.96, 0.04 + 0.2 * k), 'right', 'bottom'))
        ax.annotate(f'({b[i]:.2f}, {d[i]:.2f})\ndepth = {abs(d[i] - b[i]):.2f} hPa',
                    (b[i], d[i]), xytext=pos, textcoords='axes fraction', ha=ha, va=va,
                    fontsize=fontsize, zorder=6,
                    bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=col, lw=0.8),
                    arrowprops=dict(arrowstyle='-', color=col, lw=0.8, shrinkA=0, shrinkB=4))
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.set_xlabel('Birth (hPa)'); ax.set_ylabel('Death (hPa)')
    return lim


def _save(fig, relpath, out_dir=None):
    out = os.path.join(out_dir or FIG_DIR, relpath)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, **JPG)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Figures 1-4: Sec. 2 (synthetic)                                              #
# --------------------------------------------------------------------------- #

def fig01_torus(out_dir=None):
    """Fig. 1: hollow torus with its two non-equivalent 1-holes."""
    plt = _plt()
    Rt, rt = 2.0, 0.8
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 120), np.linspace(0, 2 * np.pi, 60))
    X = (Rt + rt * np.cos(v)) * np.cos(u); Y = (Rt + rt * np.cos(v)) * np.sin(u); Z = rt * np.sin(v)
    fig = plt.figure(figsize=(6.4, 4.4))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, color='0.82', alpha=0.55, linewidth=0, antialiased=True, shade=True)
    t = np.linspace(0, 2 * np.pi, 300)
    ax.plot((Rt + rt * np.cos(np.pi / 2)) * np.cos(t), (Rt + rt * np.cos(np.pi / 2)) * np.sin(t),
            np.full_like(t, rt), color=RED, lw=2.5, zorder=10)            # around the hole
    u0 = -np.pi / 2
    ax.plot((Rt + rt * np.cos(t)) * np.cos(u0), (Rt + rt * np.cos(t)) * np.sin(u0), rt * np.sin(t),
            color=BLUE, lw=2.5, zorder=10)                                  # around the tube
    ax.set_box_aspect((1, 1, 0.35)); ax.view_init(elev=32, azim=-60); ax.set_axis_off()
    return _save(fig, 'torus_with_loops.jpg', out_dir)


def fig02_grid(out_dir=None):
    """Fig. 2: the grayscale image and its grid of intensities (the same array)."""
    plt = _plt()
    outs = []
    for name, numbers in (('grayscale_visualization.jpg', False),
                          ('tesselation_visualization.jpg', True)):
        # the original's proportions: three 16 pt digits fill ~80 % of a 46 pt cell
        fig, ax = plt.subplots(figsize=(11.6, 5.8))
        ax.imshow(GRID, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
        if numbers:
            for (r, c), v in np.ndenumerate(GRID):
                ax.text(c, r, f'{int(v)}', ha='center', va='center', fontsize=16,
                        color='white' if v < 128 else 'black')
            for k in range(GRID.shape[1] + 1):
                ax.axvline(k - 0.5, color='k', lw=0.5)
            for k in range(GRID.shape[0] + 1):
                ax.axhline(k - 0.5, color='k', lw=0.5)
        ax.set_xticks([]); ax.set_yticks([])
        outs.append(_save(fig, os.path.join('cubical_tess', name), out_dir))
    return outs


def fig03_synthetic_filtration(out_dir=None):
    """Fig. 3: filtration strips and persistence diagrams of the Fig. 2 image.

    The strips follow Def. 2.1 exactly (sublevel: I <= t, superlevel: I >= t),
    with pixels as top cells.  The diagrams are computed, not typed.
    """
    plt = _plt()
    outs = []
    for tag in ('sup', 'sub'):
        ts = SYN_THRESHOLDS[tag]
        fig, axes = plt.subplots(len(ts), 1, figsize=(3.0, 8.0),
                                 gridspec_kw={'hspace': 0.12})
        for ax, t in zip(axes, ts):
            inside = GRID >= t if tag == 'sup' else GRID <= t
            ax.imshow(np.where(inside, 0.5, 1.0), cmap='gray', vmin=0, vmax=1,
                      interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.6); sp.set_color('0.3')
        name = 'sup_level_syn_filt.jpg' if tag == 'sup' else 'sub_level_syn_filt.jpg'
        outs.append(_save(fig, os.path.join('cubical_fil_synt', name), out_dir))

        D = h1_diagram(GRID, superlevel=(tag == 'sup'), construction='T')
        fig, ax = plt.subplots(figsize=(3.2, 3.2))
        ax.plot([0, 262], [0, 262], color='k', lw=0.8)
        ax.scatter(D[:, 0], D[:, 1], s=25, c='k', zorder=3)
        for b, d in D[:, :2]:
            ax.annotate(f'({b:.0f}, {d:.0f})', (b, d), xytext=(5, 1), textcoords='offset points',
                        fontsize=9)
        ax.set_xlim(0, 262); ax.set_ylim(0, 262); ax.set_aspect('equal')
        ax.set_xticks([0, 100, 200, 255]); ax.set_yticks([0, 100, 200, 255])
        ax.set_xlabel('Birth'); ax.set_ylabel('Death')
        name = 'pd_cub_sup_syn.jpg' if tag == 'sup' else 'pd_cub_sub_syn.jpg'
        outs.append(_save(fig, os.path.join('cubical_fil_synt', name), out_dir))
    return outs


def synthetic_tracking():
    """Diagrams and optimal matchings of the Fig. 4 sequence.

    Returns (frames, diagrams, matchings); ``diagrams[k]`` has rows
    (birth, death, row, col) sorted by persistence, and ``matchings[k]`` pairs
    frame k with frame k+1 by the L1 1-Wasserstein optimal matching.
    """
    frames = synthetic_sequence()
    dgms = [h1_diagram(f, construction='T') for f in frames]
    dgms = [D[np.abs(D[:, 1] - D[:, 0]) > 1] for D in dgms]
    matchings = []
    for D0, D1 in zip(dgms[:-1], dgms[1:]):
        _, pairs = wasserstein(D0[:, :2], D1[:, :2], ground_metric='L1', matching=True)
        matchings.append([(i, j) for i, j in pairs if i >= 0 and j >= 0])
    return frames, dgms, matchings


def fig04_synthetic_tracking(out_dir=None):
    """Fig. 4 (a) and (c).  Panel (b) is TikZ in the manuscript; its coordinates
    are written to ``cubical_fil_synt/pd_sequence_points.txt`` from the same
    computation and must match the .tex."""
    from scipy.ndimage import label
    plt = _plt()
    frames, dgms, matchings = synthetic_tracking()
    # identify each diagram point with its spot through the matching chain
    ids = [{0: 'A', 1: 'B'} if dgms[0][0, 3] < SEQ_SIZE / 2 else {0: 'B', 1: 'A'}]
    for k, pairs in enumerate(matchings):
        ids.append({j: ids[k][i] for i, j in pairs})
    colour = {'A': ORANGE, 'B': GREEN}
    dcol = {'A': '#6ec6ff', 'B': RED}
    outs = []
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), gridspec_kw={'wspace': 0.05})
    for ax, f in zip(axes, frames):
        ax.imshow(f, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])
    outs.append(_save(fig, os.path.join('cubical_fil_synt', 'sequence_gradient_spots_shrink.jpg'), out_dir))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), gridspec_kw={'wspace': 0.05})
    for ax, f, D, idmap in zip(axes, frames, dgms, ids):
        ax.imshow(f, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
        for k, (b, d, r, c) in enumerate(D):
            spot = idmap[k]
            # T-construction (pixels are top cells): holes are 4-connected
            lab, _ = label(f > b)
            comp = lab == lab[int(r), int(c)]
            # boundary of the enclosed region = a representative 1-cycle at birth
            for rr, cc in np.argwhere(comp):
                for dr, dc, seg in ((-1, 0, [(cc - .5, rr - .5), (cc + .5, rr - .5)]),
                                    (1, 0, [(cc - .5, rr + .5), (cc + .5, rr + .5)]),
                                    (0, -1, [(cc - .5, rr - .5), (cc - .5, rr + .5)]),
                                    (0, 1, [(cc + .5, rr - .5), (cc + .5, rr + .5)])):
                    r2, c2 = rr + dr, cc + dc
                    if not (0 <= r2 < f.shape[0] and 0 <= c2 < f.shape[1]) or not comp[r2, c2]:
                        ax.plot(*zip(*seg), color=colour[spot], lw=2.2, solid_capstyle='round')
            ax.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, fill=False, ec=dcol[spot], lw=2.2))
        ax.set_xticks([]); ax.set_yticks([])
    outs.append(_save(fig, os.path.join('cubical_fil_synt', 'seq_syn_rep.jpg'), out_dir))

    lines = ['# Fig. 4(b) TikZ coordinates (birth, death), computed by',
             '# src/paper_figures.py::synthetic_tracking (sublevel, T-construction).']
    for k, (D, idmap) in enumerate(zip(dgms, ids)):
        pts = {idmap[i]: (int(D[i, 0]), int(D[i, 1])) for i in range(len(D))}
        lines.append(f'frame {k + 1}: orange (spot A) {pts["A"]}   green (spot B) {pts["B"]}')
    p = os.path.join(out_dir or FIG_DIR, 'cubical_fil_synt', 'pd_sequence_points.txt')
    with open(p, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    outs.append(p)
    return outs


# --------------------------------------------------------------------------- #
# Figures 5-8: Sec. 3                                                          #
# --------------------------------------------------------------------------- #

EXAMPLE_DATE = dt.date(2023, 12, 31)


def _enclosed_region(stored, rc, level):
    """Region enclosed by a live 1-cycle at filtration level ``level`` (stored
    units, sublevel convention): the 8-connected component of {stored > level}
    containing the feature's death vertex.  By duality this is exactly the hole
    the class represents."""
    from scipy.ndimage import label
    lab, _ = label((stored > level) & DISC, structure=np.ones((3, 3)))
    return lab == lab[rc]


def fig06_filtration(out_dir=None, date=EXAMPLE_DATE):
    """Fig. 6: sub- and superlevel filtrations of the example day, with the two
    deepest features of each highlighted (orange = deepest, green = second), and
    the production (V-construction) persistence diagrams."""
    plt = _plt()
    outs = []
    levels = {'sup': [10, 0, -10, -20], 'sub': [20, 10, 0, -10]}
    for tag in ('sup', 'sub'):
        R = load_reps(date, tag)
        stored = load_disc(date, tag)
        s = -1.0 if tag == 'sup' else 1.0
        top = list(np.argsort(-R['depth'])[:2])
        cols = [ORANGE, GREEN]
        dv = [death_vertex(R, i, date, tag) for i in top]
        f = stored * s / 100.0                                   # anomaly, hPa
        for t in levels[tag]:
            inside = (f >= t) if tag == 'sup' else (f <= t)
            rgb = np.ones((N, N, 4))
            rgb[inside] = (0.62, 0.62, 0.62, 1)
            for i, col, rc in zip(top, cols, dv):
                b, d = R['birth'][i], R['death'][i]
                alive = (d < t <= b) if tag == 'sup' else (b <= t < d)
                if alive:
                    reg = _enclosed_region(stored, rc, s * t * 100.0)
                    import matplotlib.colors as mc
                    rgb[reg] = (*mc.to_rgb(col), 1)
            rgb[~DISC] = (1, 1, 1, 0)
            fig = plt.figure(figsize=(4.2, 4.2))
            ax = map_axes(fig, 111)
            ax.imshow(rgb, origin='lower', extent=EXTENT, transform=ax.projection,
                      interpolation='nearest', zorder=1)
            decorate_map(ax, lw=0.5)
            sub = 'sup_level' if tag == 'sup' else 'sub_level'
            outs.append(_save(fig, os.path.join('cubical_fil_slp', sub, f'{tag}_lvl_{t}.jpg'), out_dir))
        plt = _plt(8)
        fig, ax = plt.subplots(figsize=(3.1, 3.1))
        draw_pd(ax, R, highlight=[(i, c, 'o') for i, c in zip(top, cols)], fontsize=7, point_size=5)
        outs.append(_save(fig, os.path.join('cubical_fil_slp', f'pd_cub_{tag}.jpg'), out_dir))
    return outs


# --------------------------------------------------------------------------- #
# Figures 9-11: Sec. 4.1-4.2                                                   #
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Figures 12-15: Sec. 4.3                                                      #
# --------------------------------------------------------------------------- #


FIGURES = {
    'fig01': fig01_torus, 'fig02': fig02_grid, 'fig03': fig03_synthetic_filtration,
    'fig04': fig04_synthetic_tracking, 'fig06': fig06_filtration,
}
