"""The gallery is generated from program metadata, never hand-edited."""

import textwrap

from figlib.gallery import build_gallery

PROG = textwrap.dedent('''
    import numpy as np
    from figlib.scene import Curve, Scene
    from figlib.style import Role

    CLAIM = "The line rises monotonically."
    EXPOSITION = """{exposition}"""
    PARAMS = {{"n": 9}}

    def compute(p):
        x = np.linspace(0, 1, p["n"])
        return np.column_stack([x, x])

    def build(g):
        return Scene(items=[Curve(g, role=Role.CONTENT)])

    def assertions(g):
        assert g[-1, 1] > g[0, 1], "should rise"
''')


def _corpus(tmp_path):
    figs = tmp_path / "figures"
    (figs / "complex").mkdir(parents=True)
    (figs / "signals").mkdir()
    (figs / "complex" / "alpha.py").write_text(
        PROG.format(exposition="Alpha exposition text."))
    (figs / "signals" / "beta.py").write_text(
        PROG.format(exposition="Beta exposition text."))
    return figs


def test_gallery_groups_by_subject_in_reading_order(tmp_path):
    """Subjects are ordered by SUBJECT_ORDER, not alphabetically -- complex
    comes before signals because that is the reading order, and 'c' < 's'
    would hide a broken ordering."""
    md = build_gallery(_corpus(tmp_path))
    assert md.index("## complex") < md.index("## signals")


def test_gallery_carries_claim_and_exposition(tmp_path):
    md = build_gallery(_corpus(tmp_path))
    assert "The line rises monotonically." in md
    assert "Alpha exposition text." in md
    assert "Beta exposition text." in md


def test_gallery_links_both_grounds(tmp_path):
    """The card shows the default cream render; the transparent one is what
    people embed, so it must stay reachable."""
    md = build_gallery(_corpus(tmp_path))
    assert "complex/alpha.png" in md
    assert "complex/alpha_transparent.svg" in md


def test_gallery_marks_a_missing_exposition_rather_than_omitting_it(tmp_path):
    """A figure with no EXPOSITION must still appear, flagged -- the gallery is
    the worklist for writing them, so silently dropping one hides the gap."""
    figs = _corpus(tmp_path)
    (figs / "complex" / "bare.py").write_text(
        PROG.format(exposition="x").replace('EXPOSITION = """x"""', ""))
    md = build_gallery(figs)
    assert "bare" in md and "no EXPOSITION yet" in md


def test_gallery_reads_source_and_never_imports(tmp_path):
    """Metadata comes from ast, so listing N figures does not cost N renders
    and a figure that is mid-repair still appears."""
    figs = _corpus(tmp_path)
    (figs / "complex" / "broken.py").write_text(
        PROG.format(exposition="Broken exposition text.")
        + "\nraise RuntimeError('import would explode')\n")
    md = build_gallery(figs)
    assert "broken" in md and "Broken exposition text." in md


def test_gallery_folds_a_concatenated_claim(tmp_path):
    """Long CLAIMs in this corpus are written as ("part " "part"); the reader
    must fold them, not drop them."""
    figs = _corpus(tmp_path)
    (figs / "complex" / "joined.py").write_text(
        PROG.format(exposition="Joined exposition.")
        .replace('CLAIM = "The line rises monotonically."',
                 'CLAIM = (\n    "The line rises "\n    "monotonically."\n)'))
    md = build_gallery(figs)
    assert "The line rises monotonically." in md


# --- the README grid -------------------------------------------------------

from figlib.gallery import (CONTENT_WIDTH_PX, README_END, README_START,
                            ROW_TARGET_HEIGHT_PX, build_readme_gallery,
                            justify_rows, write_readme_gallery)


def test_justify_rows_never_exceeds_the_column_width():
    """A row wider than GitHub's README column rewraps in the browser, which
    breaks the grid exactly where the math said it was justified. Every row
    must land at or under the width, gaps included."""
    aspects = [0.88, 1.0, 1.08, 1.56, 2.24, 4.07, 1.35, 1.79, 2.1]
    for row in justify_rows(aspects):
        width = sum(aspects[i] * h for i, h, _ in row) + 5 * (len(row) - 1)
        assert width <= CONTENT_WIDTH_PX + 1, f"row overflows: {width}"


def test_justify_rows_covers_every_image_once_in_order():
    aspects = [1.0, 2.0, 3.0, 1.5, 0.9]
    seen = [i for row in justify_rows(aspects) for i, _, _ in row]
    assert seen == list(range(len(aspects)))


def test_a_short_final_row_is_not_stretched_to_fill():
    """A tall narrow figure alone on the last row would justify to 4x the
    height of the grid above it. Cap it instead."""
    rows = justify_rows([2.0, 2.0, 0.5])
    last = rows[-1]
    assert len(last) == 1, rows
    assert last[0][1] <= ROW_TARGET_HEIGHT_PX * 1.3 + 1
    assert last[0][2] is False, "a short row must be flagged, so it can centre"


def test_readme_gallery_links_thumbnails_to_the_gallery_entry(tmp_path):
    md = build_readme_gallery(_corpus(tmp_path))
    assert 'figures/out/complex/alpha.png' in md
    assert 'figures/out/GALLERY.md#alpha' in md
    assert "<table" not in md, "GitHub borders and stripes every table cell"


def test_write_readme_gallery_replaces_only_the_marked_region(tmp_path):
    figs = _corpus(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text(f"# top\n\nkeep me\n\n{README_START}\nstale\n{README_END}\n\ntail\n")
    write_readme_gallery(readme, figs)
    txt = readme.read_text()
    assert "keep me" in txt and "tail" in txt and "stale" not in txt
    assert txt.count(README_START) == 1 and txt.count(README_END) == 1
