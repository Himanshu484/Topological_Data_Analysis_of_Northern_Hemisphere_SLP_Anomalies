"""
Regression tests tying the manuscript's pedagogical figures (Figs 1-6) to the
code that draws them (src/paper_figures.py).  The tests that need
data/processed_data, or the manuscript source next to the repository, are
skipped when these are absent (neither is tracked in git).
"""
import os
import re
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import paper_figures as pf            # noqa: E402
from tda_metrics import wasserstein   # noqa: E402

MANUSCRIPT = os.environ.get('MANUSCRIPT_DIR') or os.path.join(os.path.dirname(ROOT), 'manuscript')
TEX = os.path.join(MANUSCRIPT, 'main.tex')
HAVE_DATA = os.path.isdir(pf.REP_DIR) and os.path.isdir(pf.SLP_DIR)
needs_data = pytest.mark.skipif(not HAVE_DATA, reason='data/processed_data not present')
needs_tex = pytest.mark.skipif(not os.path.exists(TEX), reason='manuscript source not present')


def _tex():
    with open(TEX) as fh:
        return fh.read()


@needs_tex
def test_every_included_figure_exists():
    paths = re.findall(r'\\includegraphics\[[^\]]*\]\{([^}]*)\}', _tex())
    missing = [p for p in paths
               if not os.path.exists(os.path.join(MANUSCRIPT, p.replace('//', '/')))]
    assert not missing, missing


def test_map_placement_matches_the_analysis_inverse_map():
    assert pf.check_projection(n=300) < 0.01


def test_fig3_diagrams_are_the_published_points_in_both_constructions():
    for c in ('T', 'V'):
        sub = pf.h1_diagram(pf.GRID, superlevel=False, construction=c)[:, :2]
        sup = pf.h1_diagram(pf.GRID, superlevel=True, construction=c)[:, :2]
        assert sub.tolist() == [[10.0, 254.0]]
        assert sup.tolist() == [[220.0, 1.0]]


@needs_tex
def test_fig3_threshold_ticks_sit_at_the_panel_centres():
    """Fig. 3's TikZ ticks are placed at fractions of the strip's height; they
    must be the centres of the strip's panels, labelled with their thresholds."""
    from PIL import Image
    tex = _tex()
    for fn, tag in (('sup_level_syn_filt.jpg', 'sup'), ('sub_level_syn_filt.jpg', 'sub')):
        i = tex.index(fn)
        block = tex[i:tex.index('\\end{tikzpicture}', i)]
        pos = [float(x) for x in re.findall(r'coordinate\[pos=([\d.]+)\]', block)]
        labels = [int(x) for x in re.findall(r'k\d/(-?\d+)', block)]
        assert labels == pf.SYN_THRESHOLDS[tag]
        im = np.asarray(Image.open(os.path.join(pf.FIG_DIR, 'cubical_fil_synt', fn)).convert('L'), float)
        col = int(np.argmax((im < 180).mean(axis=0) > 0.5))      # left frame line of the panels
        on = np.where(im[:, col] < 180)[0]
        runs = np.split(on, np.where(np.diff(on) > 1)[0] + 1)
        centres = [(r[0] + r[-1]) / 2 / (im.shape[0] - 1) for r in runs]
        assert len(centres) == len(pos) == len(labels)
        assert np.allclose(centres, pos, atol=0.003)


@needs_tex
def test_fig4_tikz_coordinates_are_the_computed_diagrams():
    """Fig. 4(b) is TikZ in the .tex: its points must be the diagrams of 4(a)."""
    _, dgms, matchings = pf.synthetic_tracking()
    # both 1-holes are born on the uniform background
    for D in dgms:
        assert len(D) == 2 and np.all(D[:, 0] == pf.SEQ_BG)
    # the optimal matching follows each spot
    for pairs in matchings:
        assert sorted(pairs) == [(0, 0), (1, 1)]
    tex = _tex()
    block = tex[tex.index('\\subfloat[Sequence of persistence diagrams]'):
                tex.index('\\subfloat[Sequence of grayscale images with representative 1-cycles]')]
    typed = [tuple(map(int, m)) for m in re.findall(r'coordinates\s*\{\((\d+),(\d+)\)\}', block)]
    computed = []
    for D in dgms:                      # TikZ order per frame: orange (spot A), then green (spot B)
        a, b = (D[0], D[1]) if D[0, 3] < pf.SEQ_SIZE / 2 else (D[1], D[0])
        computed += [(int(a[0]), int(a[1])), (int(b[0]), int(b[1]))]
    assert typed == computed


def test_wasserstein_is_invariant_under_the_superlevel_sign_flip():
    rng = np.random.default_rng(1)
    for _ in range(50):
        A = np.sort(rng.normal(0, 10, (rng.integers(0, 6), 2)), axis=1)
        B = np.sort(rng.normal(0, 10, (rng.integers(0, 6), 2)), axis=1)
        w = wasserstein(A, B, ground_metric='L1')
        assert w >= 0
        assert np.isclose(w, wasserstein(-A, -B, ground_metric='L1'))


@needs_tex
@needs_data
def test_fig6_depths_quoted_in_the_text():
    tex = _tex()
    for tag, quoted in (('sup', ('22.50', '29.34')), ('sub', ('11.89', '21.75'))):
        R = pf.load_reps(pf.EXAMPLE_DATE, tag)
        top = np.sort(R['depth'])[-2:]
        assert [f'{v:.2f}' for v in top] == list(quoted)
        for q in quoted:
            assert q in tex
