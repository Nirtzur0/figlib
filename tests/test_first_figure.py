"""The README's quickstart is a real program, and stays one.

`docs/examples/first_figure.py` is quoted verbatim in the README. It lives
outside `figures/`, so `make regress` never sees it — this is the only thing
standing between the landing page and fiction.
"""

from pathlib import Path

from figlib.program import run

EXAMPLE = Path(__file__).parent.parent / "docs" / "examples" / "first_figure.py"


def test_readme_quickstart_passes_every_gate(tmp_path):
    report = run(EXAMPLE, out_dir=tmp_path)
    assert report.passed, report.summary()
