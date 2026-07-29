"""End-to-end: a toy figure program runs through the full pipeline."""

import textwrap

from figlib.program import run


TOY = textwrap.dedent('''
    import numpy as np
    from figlib.scene import Curve, MathLabel, Scene
    from figlib.style import Role

    CLAIM = "The unit parabola passes through the origin."
    PARAMS = {"n": 50}

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
