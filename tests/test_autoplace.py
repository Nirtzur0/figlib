"""The auto-place pass: free-nudge resolution promoted from gate
diagnostic to deterministic pre-gate solver. Unsolvable layouts must
fall through to the mechanical gate unchanged."""

import textwrap

import numpy as np

from figlib.autoplace import MAX_AUTO_NUDGE, autoplace, autoplace_figure
from figlib.figure import Figure, Panel
from figlib.gates import mechanical, mechanical_figure
from figlib.scene import Brace, Curve, MathLabel, Scene
from figlib.style import DEFAULT_STYLE, Role


def _kinds(diags):
    return {d.kind for d in diags}


def _base_scene(*labels):
    return Scene(items=[Curve(np.array([[0.0, 0.0], [1.0, 1.0]]),
                              role=Role.CONTENT), *labels],
                 xlim=(0, 1), ylim=(0, 1))


class TestAutoplaceScene:
    def test_colliding_labels_resolved(self):
        a = MathLabel(r"e^{i\theta}", (0.5, 0.5))
        b = MathLabel(r"e^{i\phi}", (0.5, 0.5))
        scene = _base_scene(a, b)
        assert "label-collision" in _kinds(mechanical(scene, DEFAULT_STYLE))

        notes = autoplace(scene, DEFAULT_STYLE)

        assert "label-collision" not in _kinds(mechanical(scene, DEFAULT_STYLE))
        assert a.offset_px != (0.0, 0.0) or b.offset_px != (0.0, 0.0)
        assert notes  # the applied moves are reported

    def test_clean_scene_untouched(self):
        a = MathLabel(r"x", (0.2, 0.6))
        b = MathLabel(r"y", (0.7, 0.2))
        scene = _base_scene(a, b)

        notes = autoplace(scene, DEFAULT_STYLE)

        assert notes == []
        assert a.offset_px == (0.0, 0.0) and b.offset_px == (0.0, 0.0)

    def test_pinned_labels_never_move(self):
        a = MathLabel(r"e^{i\theta}", (0.5, 0.5), pin=True)
        b = MathLabel(r"e^{i\phi}", (0.5, 0.5), pin=True)
        scene = _base_scene(a, b)

        autoplace(scene, DEFAULT_STYLE)

        assert a.offset_px == (0.0, 0.0) and b.offset_px == (0.0, 0.0)
        assert "label-collision" in _kinds(mechanical(scene, DEFAULT_STYLE))

    def test_deterministic_across_runs(self):
        def build():
            return _base_scene(MathLabel(r"\alpha", (0.5, 0.5)),
                               MathLabel(r"\beta", (0.5, 0.5)),
                               MathLabel(r"\gamma", (0.52, 0.5)))

        s1, s2 = build(), build()
        autoplace(s1, DEFAULT_STYLE)
        autoplace(s2, DEFAULT_STYLE)

        offs1 = [it.offset_px for it in s1.items if isinstance(it, MathLabel)]
        offs2 = [it.offset_px for it in s2.items if isinstance(it, MathLabel)]
        assert offs1 == offs2

    def test_clipped_label_pulled_inside(self):
        # anchored on the top edge, nudged out of frame by its offset
        a = MathLabel(r"x", (0.5, 1.0), offset_px=(0.0, -65.0))
        scene = _base_scene(a)
        assert "clipped" in _kinds(mechanical(scene, DEFAULT_STYLE))

        autoplace(scene, DEFAULT_STYLE)

        assert "clipped" not in _kinds(mechanical(scene, DEFAULT_STYLE))

    def test_oversize_overlap_left_to_gate(self):
        # coincident labels whose every free nudge exceeds the budget:
        # the solver must leave them for the gate, not teleport them
        big = r"\frac{x^2+1}{y^2+1}"
        a = MathLabel(big, (0.5, 0.5), size_pt=30)
        b = MathLabel(big, (0.5, 0.5), size_pt=30)
        scene = _base_scene(a, b)

        autoplace(scene, DEFAULT_STYLE)

        assert a.offset_px == (0.0, 0.0) and b.offset_px == (0.0, 0.0)
        assert "label-collision" in _kinds(mechanical(scene, DEFAULT_STYLE))

    def test_brace_label_is_an_obstacle_not_movable(self):
        # a MathLabel colliding with a brace's label: only the MathLabel
        # can move (the brace label is derived from brace geometry)
        a = MathLabel(r"w", (0.5, 0.58))
        scene = Scene(items=[
            Curve(np.array([[0.0, 0.0], [1.0, 1.0]]), role=Role.CONTENT),
            Brace((0.0, 0.5), (1.0, 0.5), side=1.0, label=r"L"),
            a,
        ], xlim=(0, 1), ylim=(0, 1))
        assert "label-collision" in _kinds(mechanical(scene, DEFAULT_STYLE))

        autoplace(scene, DEFAULT_STYLE)

        assert "label-collision" not in _kinds(mechanical(scene, DEFAULT_STYLE))
        assert a.offset_px != (0.0, 0.0)


class TestAutoplaceFigure:
    def test_page_items_resolved_figure_wide(self):
        fig = Figure(
            panels=[Panel(_base_scene())],
            page_items=[MathLabel(r"\mu", (120.0, 60.0)),
                        MathLabel(r"\nu", (120.0, 60.0))],
        )
        assert "label-collision" in _kinds(mechanical_figure(fig, DEFAULT_STYLE))

        notes = autoplace_figure(fig, DEFAULT_STYLE)

        assert "label-collision" not in _kinds(mechanical_figure(fig, DEFAULT_STYLE))
        assert notes


COLLIDING_PROGRAM = textwrap.dedent('''
    import numpy as np
    from figlib.scene import Curve, MathLabel, Scene
    from figlib.style import Role

    CLAIM = "Two labels want the same spot; the solver separates them."
    PARAMS = {}

    def compute(p):
        x = np.linspace(0, 1, 20)
        return np.column_stack([x, x])

    def build(g):
        return Scene(items=[
            Curve(g, role=Role.CONTENT),
            MathLabel(r"f", (0.5, 0.5)),
            MathLabel(r"g", (0.5, 0.5)),
        ], xlim=(0, 1), ylim=(0, 1))

    def assertions(g):
        pass
''')


class TestProgramIntegration:
    def test_run_autoplaces_before_gates(self, tmp_path):
        from figlib.program import run
        prog = tmp_path / "toy_colliding.py"
        prog.write_text(COLLIDING_PROGRAM)

        report = run(prog, out_dir=tmp_path / "out")
        assert report.passed, report.summary()
        assert report.notes  # the summary carries what moved

    def test_run_place_false_leaves_collision(self, tmp_path):
        from figlib.program import run
        prog = tmp_path / "toy_colliding2.py"
        prog.write_text(COLLIDING_PROGRAM)

        report = run(prog, out_dir=tmp_path / "out", place=False)
        assert any(d.kind == "label-collision" for d in report.diagnostics)
