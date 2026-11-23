"""
Where the paper scripts write their tables and figures.

The manuscript is kept outside this repository.  The scripts write into

    $MANUSCRIPT_DIR            if the environment variable is set, otherwise
    ../manuscript              if that folder exists next to the repository, otherwise
    ./paper_outputs            inside the repository (created on demand),

in the sub-folders tables/ and figures/.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _manuscript():
    env = os.environ.get('MANUSCRIPT_DIR')
    if env:
        return os.path.abspath(env)
    sibling = os.path.join(os.path.dirname(ROOT), 'manuscript')
    return sibling if os.path.isdir(sibling) else os.path.join(ROOT, 'paper_outputs')


MANUSCRIPT = _manuscript()
TABLES = os.path.join(MANUSCRIPT, 'tables')
FIGURES = os.path.join(MANUSCRIPT, 'figures')
NULL_TABLES = os.path.join(ROOT, 'output', 'paper', 'null_tables')      # intermediate fragments, see make_null_tables.py
for _d in (TABLES, FIGURES, NULL_TABLES):
    os.makedirs(_d, exist_ok=True)
