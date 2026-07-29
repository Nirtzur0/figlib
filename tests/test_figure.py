"""Multi-panel Figure layer: slot geometry, connector glyphs, figure-wide
mechanical gate, program dispatch."""

import textwrap

import numpy as np
import pytest

from figlib.figure import (Connector, Figure, Panel, connector_ink,
                           layout_figure)
from figlib.gates import mechanical, mechanical_figure
from figlib.render import figure_to_svg
from figlib.scene import Curve, MathLabel, Scene
from figlib.style import DEFAULT_STYLE


def square_scene(**kw) -> Scene:
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    return Scene(items=[Curve(pts, closed=True), *kw.pop("extra", [])])


def two_panel(gap: float = 46.0, **fig_kw) -> Figure:
    return Figure(panels=[Panel(square_scene(), tag="[a]"),
                          Panel(square_scene(), tag="[b]")],
                  gap_px=gap, **fig_kw)


class TestLayout:
    def test_two_equal_slots_and_gap_sum_to_width(self):
        lay = layout_figure(two_panel(), width_px=900)
        s0, s1 = lay.slots
        assert s0.w == pytest.approx(s1.w)
        assert s0.w + 46 + s1.w == pytest.approx(900)
        assert s1.x == pytest.approx(s0.w + 46)
        assert lay.canvas_w == 900

    def test_width_frac_honored(self):
        fig = Figure(panels=[Panel(square_scene(), width_frac=0.3),
                             Panel(square_scene())])
        lay = layout_figure(fig, width_px=900)
        avail = 900 - 46
        assert lay.slots[0].w == pytest.approx(0.3 * avail)
        assert lay.slots[1].w == pytest.approx(0.7 * avail)

    def test_grid_fills_row_major(self):
        fig = Figure(panels=[Panel(square_scene()) for _ in range(4)],
                     grid=(2, 2))
        lay = layout_figure(fig, width_px=900)
        assert lay.slots[0].y == lay.slots[1].y
        assert lay.slots[2].y == lay.slots[3].y
        assert lay.slots[2].y > lay.slots[0].y
        assert lay.slots[0].x == lay.slots[2].x
        assert lay.canvas_h == pytest.approx(
            lay.slots[0].h + 46 + lay.slots[2].h)

    def test_short_panel_vertically_centered(self):
        tall = Scene(items=[Curve(np.array([[0.0, 0.0], [1.0, 2.0]]))],
                     xlim=(0, 1), ylim=(0, 2))
        fig = Figure(panels=[Panel(tall), Panel(square_scene())])
        lay = layout_figure(fig, width_px=900)
        s0, s1 = lay.slots
        assert s0.h == s1.h  # row height is the max
        assert s1.transform.canvas_h < s1.h
        assert s1.content_y == pytest.approx((s1.h - s1.transform.canvas_h) / 2)


class TestConnector:
    def test_map_glyph_lives_in_the_gap(self):
        fig = two_panel()
        fig.connectors = [Connector(0, 1, kind="map", label=r"z^{2}")]
        lay = layout_figure(fig, width_px=900)
        ink = connector_ink(fig.connectors[0], lay, DEFAULT_STYLE)
        gap_x0, gap_x1 = lay.slots[0].w, lay.slots[1].x
        all_x = np.concatenate([p[:, 0] for p in ink.strokes + ink.heads])
        assert all_x.min() >= gap_x0 and all_x.max() <= gap_x1
        # label sits at the gap's horizontal midpoint
        assert ink.labels[0].anchor[0] == pytest.approx((gap_x0 + gap_x1) / 2)

    def test_double_map_has_two_strokes_opposed(self):
        fig = two_panel()
        fig.connectors = [Connector(0, 1, double=True)]
        lay = layout_figure(fig, width_px=900)
        ink = connector_ink(fig.connectors[0], lay, DEFAULT_STYLE)
        assert len(ink.strokes) == 2 and len(ink.heads) == 2
        fwd = ink.strokes[0][-1, 0] - ink.strokes[0][0, 0]
        back = ink.strokes[1][-1, 0] - ink.strokes[1][0, 0]
        assert fwd > 0 > back

    def test_eq_glyph_is_a_label_not_ink(self):
        fig = two_panel()
        fig.connectors = [Connector(0, 1, kind="eq")]
        lay = layout_figure(fig, width_px=900)
        ink = connector_ink(fig.connectors[0], lay, DEFAULT_STYLE)
        assert not ink.strokes and not ink.heads and not ink.hollow
        assert ink.labels[0].latex == "="

    def test_block_arrow_is_hollow_polygon(self):
        fig = two_panel()
        fig.connectors = [Connector(0, 1, kind="block")]
        lay = layout_figure(fig, width_px=900)
        ink = connector_ink(fig.connectors[0], lay, DEFAULT_STYLE)
        assert len(ink.hollow) == 1 and ink.hollow[0].shape == (7, 2)


class TestFigureRender:
    def test_svg_has_translated_panel_groups_and_frames(self):
        fig = two_panel()
        fig.connectors = [Connector(0, 1, label=r"f")]
        svg = figure_to_svg(fig, DEFAULT_STYLE, width_px=900)
        assert svg.count('<g transform="translate(') == 2
        assert svg.count('rx="10.00"') == 2  # two rounded frames
        assert "arrowhead" in svg            # squiggle's explicit head


class TestMechanicalFigure:
    def test_cross_panel_collision_caught_in_page_coords(self):
        # Each panel passes alone; their labels meet in the gap.
        a = square_scene(extra=[MathLabel(r"e^{i\theta}", (1.0, 0.5),
                                          ha="left", offset_px=(48, 0))])
        b = square_scene(extra=[MathLabel(r"e^{i\phi}", (0.0, 0.5),
                                          ha="left", offset_px=(-55, 0))])
        fig = Figure(panels=[Panel(a, frame=False), Panel(b, frame=False)])
        diags = mechanical_figure(fig, DEFAULT_STYLE, width_px=900)
        assert any(d.kind == "label-collision" for d in diags)
        assert not any(d.kind == "clipped" for d in diags)

    def test_page_item_clipping_against_figure_canvas(self):
        fig = two_panel()
        fig.page_items = [MathLabel(r"x", (-30.0, 10.0), ha="left", va="top")]
        diags = mechanical_figure(fig, DEFAULT_STYLE, width_px=900)
        assert any(d.kind == "clipped" for d in diags)

    def test_clean_figure_passes(self):
        fig = two_panel()
        fig.connectors = [Connector(0, 1, kind="map", label=r"f")]
        assert mechanical_figure(fig, DEFAULT_STYLE, width_px=900) == []

    def test_single_panel_matches_scene_gate_semantics(self):
        scene = square_scene(extra=[
            MathLabel(r"e^{i\theta}", (0.5, 0.5)),
            MathLabel(r"e^{i\phi}", (0.5, 0.5))])
        fig = Figure(panels=[Panel(scene)])
        kinds_fig = {d.kind for d in mechanical_figure(fig, DEFAULT_STYLE, 900)}
        kinds_scene = {d.kind for d in mechanical(scene, DEFAULT_STYLE, 900)}
        assert "label-collision" in kinds_fig and kinds_fig == kinds_scene


FIGURE_TOY = textwrap.dedent('''
    import numpy as np
    from figlib.figure import Connector, Figure, Panel
    from figlib.scene import Curve, Scene

    CLAIM = "The identity map sends the square to itself."
    EXPOSITION = """A synthetic figure used to exercise the runner. The
    exposition gate has a word floor, so a fixture needs real sentences
    here rather than a stub -- which is the gate working as intended, and
    the cheapest possible demonstration that it fires on prose that is
    merely present rather than actually written."""
    PARAMS = {}

    def compute(p):
        return np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

    def build(g):
        panel = lambda: Panel(Scene(items=[Curve(g, closed=True)]))
        return Figure(panels=[panel(), panel()],
                      connectors=[Connector(0, 1, label=r"\\mathrm{id}")])

    def assertions(g):
        assert g.shape == (4, 2)
''')


def test_program_run_dispatches_figure_builds(tmp_path):
    from figlib.program import run
    prog = tmp_path / "toy_panels.py"
    prog.write_text(FIGURE_TOY)
    report = run(prog, out_dir=tmp_path / "out")
    assert report.passed, report.summary()
    svg = report.svg_path.read_text()
    assert svg.count('<g transform="translate(') == 2
    assert report.png_path.exists()
