"""End-to-end: a toy figure program runs through the full pipeline."""

import textwrap

from figlib.program import run


TOY = textwrap.dedent('''
    import numpy as np
    from figlib.scene import Curve, MathLabel, Scene
    from figlib.style import Role

    CLAIM = "The unit parabola passes through the origin."
    PARAMS = {"n": 51}  # odd count so the grid samples x = 0 exactly

    def compute(p):
        x = np.linspace(-1, 1, p["n"])
        return np.column_stack([x, x**2])

    def build(g):
        return Scene(items=[
            Curve(g, role=Role.CONTENT),
            MathLabel(r"y = x^2", (0.35, 0.75), ha="left"),
        ])

    def assertions(g):
        assert abs(g[:, 1].min()) < 1e-12, "parabola should touch zero"
''')


def test_toy_program_end_to_end(tmp_path):
    prog = tmp_path / "toy_parabola.py"
    prog.write_text(TOY)
    report = run(prog, out_dir=tmp_path / "out")
    assert report.passed, report.summary()
    assert report.svg_path.exists() and report.png_path.exists()


def test_broken_assertion_reported(tmp_path):
    prog = tmp_path / "toy_broken.py"
    prog.write_text(TOY.replace("< 1e-12", "> 1.0"))
    report = run(prog, out_dir=tmp_path / "out")
    assert not report.passed
    assert any(d.kind == "numerical" for d in report.diagnostics)


# --- EXPOSITION: the argument a figure serves ------------------------------

_BARE = '''
import numpy as np
from figlib.scene import Curve, Scene
from figlib.style import Role

CLAIM = "The line rises."
PARAMS = {"n": 9}

def compute(p):
    x = np.linspace(0, 1, p["n"])
    return np.column_stack([x, x])

def build(g):
    return Scene(items=[Curve(g, role=Role.CONTENT)])

def assertions(g):
    assert g[-1, 1] > g[0, 1], "should rise"
'''


def _prog(tmp_path, extra=""):
    from figlib.program import load_program
    p = tmp_path / "bare.py"
    p.write_text(_BARE + extra)
    return load_program(p)


def test_exposition_gate_is_silent_when_disabled(tmp_path):
    """It ships disabled so the gate can land without reddening a corpus that
    has not written the field yet."""
    from figlib.gates import exposition_gate
    assert exposition_gate(_prog(tmp_path)) == []


def test_exposition_gate_fires_on_a_missing_field(tmp_path):
    from figlib.gates import exposition_gate
    diags = exposition_gate(_prog(tmp_path), enabled=True)
    assert [d.kind for d in diags] == ["exposition"]
    assert "EXPOSITION" in diags[0].detail


def test_exposition_gate_rejects_a_stub(tmp_path):
    """A one-liner is a restated CLAIM, not the passage the figure serves --
    the word floor is what stops the field decaying into a second caption."""
    from figlib.gates import exposition_gate
    mod = _prog(tmp_path, '\nEXPOSITION = "It shows a line going up."\n')
    diags = exposition_gate(mod, enabled=True)
    assert [d.kind for d in diags] == ["exposition"]
    assert "40" in diags[0].detail


def test_exposition_gate_accepts_a_real_passage(tmp_path):
    from figlib.gates import exposition_gate
    words = " ".join(["mechanism"] * 45)
    mod = _prog(tmp_path, f'\nEXPOSITION = """{words}"""\n')
    assert exposition_gate(mod, enabled=True) == []


def test_report_carries_the_exposition(tmp_path):
    """figcheck and the gallery both read it off the Report rather than
    re-parsing the program."""
    from figlib.program import run
    words = " ".join(["mechanism"] * 45)
    p = tmp_path / "withexp.py"
    p.write_text(_BARE + f'\nEXPOSITION = """{words}"""\n')
    report = run(p, out_dir=tmp_path / "out")
    assert report.exposition and report.exposition.split()[0] == "mechanism"
