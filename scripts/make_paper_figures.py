"""
Regenerate the pedagogical figures of the manuscript (Section 2 and the
example-day filtration of Section 3).

    python scripts/make_paper_figures.py                 # all five
    python scripts/make_paper_figures.py fig04 fig06     # a subset
    python scripts/make_paper_figures.py --out-dir /tmp/figs

    fig01  torus with its two loops            fig04  synthetic sequence and its tracking
    fig02  image as a cubical complex          fig06  filtrations and diagrams of one day (needs
    fig03  filtration of a synthetic image            data/processed_data/{SLP_data_years,representative_data})

Figures are written into <manuscript>/figures/ under the file names the
manuscript includes (see src/paper_figures.py).  Before drawing anything the
script checks that map placement agrees with the analysis' own pixel ->
(lat, lon) inverse to within 0.01 pixel, and it prints the numbers the
captions and the text quote.
"""
import argparse
import datetime as dt
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import paper_figures as pf   # noqa: E402


def report(name):
    """Numbers each figure's caption or the text depends on."""
    if name == 'fig04':
        _, dgms, _ = pf.synthetic_tracking()
        for k, D in enumerate(dgms):
            print(f'    frame {k + 1}: ' + ', '.join(f'({b:.0f}, {d:.0f})' for b, d in D[:, :2]))
    elif name == 'fig06':
        for tag in ('sup', 'sub'):
            R = pf.load_reps(pf.EXAMPLE_DATE, tag)
            top = np.argsort(-R['depth'])[:2]
            print(f'    {pf.LABEL[tag]}: TP1 {R["depth"].sum():.1f} hPa, TP2 {(R["depth"] ** 2).sum():.1f} hPa^2; '
                  + '; '.join(f'{c} (b {R["birth"][i]:+.2f}, d {R["death"][i]:+.2f}) depth {R["depth"][i]:.2f}'
                              for i, c in zip(top, ('orange', 'green'))))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('figures', nargs='*', help=f'subset of {sorted(pf.FIGURES)}')
    ap.add_argument('--out-dir', default=None, help='write here instead of the manuscript figures/ dir')
    args = ap.parse_args()
    names = args.figures or list(pf.FIGURES)
    unknown = set(names) - set(pf.FIGURES)
    if unknown:
        ap.error(f'unknown figure(s) {sorted(unknown)}; choose from {sorted(pf.FIGURES)}')

    err = pf.check_projection()
    print(f'map placement check: max discrepancy {err:.2e} px')
    for name in names:
        t0 = time.time()
        out = pf.FIGURES[name](out_dir=args.out_dir)
        files = out[0] if isinstance(out, tuple) else out
        files = files if isinstance(files, list) else [files]
        print(f'{name}: {len(files)} file(s) in {time.time() - t0:.0f}s')
        report(name)


if __name__ == '__main__':
    main()
