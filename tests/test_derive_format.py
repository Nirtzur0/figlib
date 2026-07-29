"""Format derivation from the annotation census.

The sizing mechanism these tests pin down: label px are absolute per
format (x ink_scale), so MARGIN -> COLUMN genuinely quarters the load
fraction, while COLUMN -> WIDE is near-invariant BY DESIGN (WIDE's
ink_scale 1.45 ~ the width ratio 1000/680 — reading-size parity). The
derivation therefore *evaluates* each candidate rather than assuming
bigger = relief, and the load gate's advice becomes a computed verdict.
"""

import textwrap

import numpy as np
import pytest

from figlib.format import (COLUMN, MARGIN, WIDE, annotation_census,
                           derive_format, smallest_carrier)
from figlib.scene import Curve, MathLabel, Scene
from figlib.style import DEFAULT_STYLE, Role

BIG = r"\frac{x^2+1}{y^2+1}"


def _scene(labels):
    return Scene(items=[Curve(np.array([[0.0, 0.0], [1.0, 1.0]]),
                              role=Role.CONTENT), *labels],
                 xlim=(0, 1), ylim=(0, 1))


def _light_scene():
    return _scene([MathLabel(r"x", (0.2, 0.6)), MathLabel(r"y", (0.7, 0.2))])


def _loaded_scene(n, size_pt, cols=6):
    xs = np.linspace(0.08, 0.92, cols)
    ys = np.linspace(0.08, 0.92, -(-n // cols))
    labels = [MathLabel(BIG, (xs[i % cols], ys[i // cols]), size_pt=size_pt,
                        ha="center", va="center")
              for i in range(n)]
    return _scene(labels)


# fails annotation-load at every format (load fraction ~invariant past COLUMN)
def _overloaded_scene():
    return _loaded_scene(30, size_pt=26)


# fails at MARGIN, fits at COLUMN (the one rung with real relief)
def _margin_overload_scene():
    return _loaded_scene(12, size_pt=20, cols=4)


class TestCensus:
    def test_margin_to_column_quarters_the_load(self):
        scene = _loaded_scene(8, size_pt=14)
        area_m, _ = annotation_census(scene, DEFAULT_STYLE, MARGIN)
        area_c, _ = annotation_census(scene, DEFAULT_STYLE, COLUMN)
        assert area_m / area_c == pytest.approx(4.0, rel=1e-3)

    def test_column_to_wide_is_near_invariant_by_design(self):
        # ink_scale parity: WIDE buys canvas and ink together, not headroom
        scene = _loaded_scene(8, size_pt=14)
        area_c, _ = annotation_census(scene, DEFAULT_STYLE, COLUMN)
        area_w, _ = annotation_census(scene, DEFAULT_STYLE, WIDE)
        assert 0.9 < area_c / area_w < 1.15


class TestSmallestCarrier:
    def test_light_scene_carried_by_column(self):
        assert smallest_carrier(_light_scene(), DEFAULT_STYLE) is COLUMN

    def test_overload_carried_by_nothing(self):
        assert smallest_carrier(_overloaded_scene(), DEFAULT_STYLE) is None

    def test_margin_overload_carried_by_column(self):
        assert smallest_carrier(_margin_overload_scene(), DEFAULT_STYLE,
                                ladder=(COLUMN, WIDE)) is COLUMN


class TestDeriveFormat:
    def test_defaults_to_column(self):
        assert derive_format(_light_scene(), DEFAULT_STYLE) is COLUMN

    def test_falls_back_to_column_when_nothing_carries(self):
        # render at the default anyway; the gate will say why, precisely
        assert derive_format(_overloaded_scene(), DEFAULT_STYLE) is COLUMN


_PROGRAM = textwrap.dedent('''
    import numpy as np
    from figlib.scene import Curve, MathLabel, Scene
    from figlib.style import Role
    {format_line}

    CLAIM = "Annotation load decides the slot."
    EXPOSITION = """A synthetic figure used to exercise the runner. The
    exposition gate has a word floor, so a fixture needs real sentences
    here rather than a stub -- which is the gate working as intended, and
    the cheapest possible demonstration that it fires on prose that is
    merely present rather than actually written."""
    PARAMS = {{}}

    def compute(p):
        x = np.linspace(0, 1, 20)
        return np.column_stack([x, x])

    def build(g):
        labels = {labels}
        return Scene(items=[Curve(g, role=Role.CONTENT), *labels],
                     xlim=(0, 1), ylim=(0, 1))

    def assertions(g):
        pass
''')

_LIGHT = '[MathLabel(r"x", (0.2, 0.6)), MathLabel(r"y", (0.7, 0.2))]'
_HEAVY = ('[MathLabel(r"\\frac{x^2+1}{y^2+1}", (0.08 + 0.14 * (i % 6), '
          '0.08 + 0.14 * (i // 6)), size_pt=26, ha="center", va="center") '
          'for i in range(30)]')
_MARGIN_HEAVY = ('[MathLabel(r"\\frac{x^2+1}{y^2+1}", (0.08 + 0.28 * (i % 4), '
                 '0.08 + 0.28 * (i // 4)), size_pt=20, ha="center", va="center") '
                 'for i in range(12)]')


def _write(tmp_path, name, format_line, labels):
    prog = tmp_path / name
    prog.write_text(_PROGRAM.format(format_line=format_line, labels=labels))
    return prog


class TestProgramIntegration:
    def test_undeclared_format_is_derived(self, tmp_path):
        from figlib.program import run
        prog = _write(tmp_path, "toy_derive.py", "", _LIGHT)
        report = run(prog, out_dir=tmp_path / "out")
        assert report.passed, report.summary()
        assert report.width_px == COLUMN.display_width_px
        assert any("derived" in n for n in report.notes)

    def test_declared_format_is_respected_not_derived(self, tmp_path):
        from figlib.program import run
        prog = _write(tmp_path, "toy_declared.py",
                      "from figlib.format import WIDE\nFORMAT = WIDE", _LIGHT)
        report = run(prog, out_dir=tmp_path / "out")
        assert report.width_px == WIDE.display_width_px
        assert not any("derived" in n for n in report.notes)

    def test_uncarriable_load_gets_computed_verdict(self, tmp_path):
        from figlib.program import run
        prog = _write(tmp_path, "toy_heavy.py", "", _HEAVY)
        report = run(prog, out_dir=tmp_path / "out")
        loads = [d for d in report.diagnostics if d.kind == "annotation-load"]
        assert loads and "no format" in loads[0].detail

    def test_declared_margin_overload_names_the_carrier(self, tmp_path):
        from figlib.program import run
        prog = _write(tmp_path, "toy_margin.py",
                      "from figlib.format import MARGIN\nFORMAT = MARGIN",
                      _MARGIN_HEAVY)
        report = run(prog, out_dir=tmp_path / "out")
        loads = [d for d in report.diagnostics if d.kind == "annotation-load"]
        assert loads and "FORMAT = COLUMN" in loads[0].detail
