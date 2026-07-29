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
    """The card shows cream because it reads better as a card; the transparent
    render is the one people embed, so it must stay reachable."""
    md = build_gallery(_corpus(tmp_path))
    assert "complex/alpha_paper.png" in md
    assert "complex/alpha.svg" in md


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
