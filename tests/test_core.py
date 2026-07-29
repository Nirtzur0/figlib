"""Core tests: typeset metrics, layout transform, SVG/PNG emission."""

import numpy as np
import pytest

from figlib.style import DEFAULT_STYLE, Role
from figlib.scene import Curve, MathLabel, Point, RightAngleMark, Scene, Vector
from figlib.typeset import render_math
from figlib.layout import Transform, geometry_extents
from figlib.render import to_svg, save


def make_scene() -> Scene:
    theta = np.linspace(0, 2 * np.pi, 100)
    circle = np.column_stack([np.cos(theta), np.sin(theta)])
    return Scene(
        items=[
            Curve(circle, role=Role.CONSTRUCTION),
            Vector((0.0, 0.0), (np.cos(0.7), np.sin(0.7)), role=Role.CONTENT),
            Point((1.0, 0.0), role=Role.CONTENT),
            MathLabel(r"e^{i\theta}", (0.75, 0.7), role=Role.CONTENT),
            RightAngleMark((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), size=0.08),
        ]
    )


class TestTypeset:
    def test_metrics_positive(self):
        lab = render_math(r"e^{i\theta}", size_pt=11)
        assert lab.width_pt > 0
        assert lab.height_pt > 0

    def test_wider_expression_is_wider(self):
        a = render_math(r"x", size_pt=11)
        b = render_math(r"x + y + z", size_pt=11)
        assert b.width_pt > a.width_pt


class TestLayout:
    def test_geometry_extents_cover_curve(self):
        scene = make_scene()
        (x0, x1), (y0, y1) = geometry_extents(scene)
        assert x0 == pytest.approx(-1.0, abs=1e-2) and x1 == pytest.approx(1.0, abs=1e-2)
        assert y0 == pytest.approx(-1.0, abs=1e-2) and y1 == pytest.approx(1.0, abs=1e-2)

    def test_transform_round_trip_and_yflip(self):
        scene = make_scene()
        t = Transform(scene, width_px=900)
        px0, py0 = t.to_canvas((0.0, 1.0))
        px1, py1 = t.to_canvas((0.0, -1.0))
        assert px0 == pytest.approx(px1)
        assert py0 < py1  # math +y is up, canvas +y is down

    def test_equal_aspect(self):
        scene = make_scene()
        t = Transform(scene, width_px=900)
        ax, _ = t.to_canvas((1.0, 0.0))
        bx, _ = t.to_canvas((0.0, 0.0))
        _, cy = t.to_canvas((0.0, 1.0))
        _, dy = t.to_canvas((0.0, 0.0))
        assert abs(ax - bx) == pytest.approx(abs(cy - dy))


class TestRender:
    def test_svg_contains_expected_elements(self):
        svg = to_svg(make_scene(), DEFAULT_STYLE, width_px=900)
        assert svg.startswith("<svg") or svg.startswith("<?xml")
        assert "<path" in svg
        assert "polygon" in svg or "arrowhead" in svg  # explicit vector head

    def test_save_writes_svg_and_png(self, tmp_path):
        stem = tmp_path / "smoke"
        save(make_scene(), stem)
        svg_file = stem.with_suffix(".svg")
        png_file = stem.with_suffix(".png")
        assert svg_file.exists() and svg_file.stat().st_size > 500
        assert png_file.exists() and png_file.stat().st_size > 500

    def test_transparent_theme_emits_no_background(self):
        from figlib.theme import RISO, RISO_T
        opaque = to_svg(make_scene(), RISO, width_px=900)
        clear = to_svg(make_scene(), RISO_T, width_px=900)
        assert "<rect" in opaque and "url(#grain)" in opaque
        assert "<rect" not in clear and "grain" not in clear


class TestCurveMarkers:
    def _line_scene(self, **curve_kw) -> Scene:
        pts = np.array([[0.0, 0.0], [10.0, 0.0]])
        return Scene(items=[Curve(pts, **curve_kw)], xlim=(0, 10), ylim=(-1, 1))

    def test_marker_at_midpoint_points_forward(self):
        import xml.etree.ElementTree as ET
        svg = to_svg(self._line_scene(arrows=(0.5,)), DEFAULT_STYLE, width_px=900)
        root = ET.fromstring(svg)
        heads = [e for e in root.iter() if e.get("class") == "arrowhead"]
        assert len(heads) == 1
        xy = np.array([[float(v) for v in p.split(",")]
                       for p in heads[0].get("points").split()])
        t = Transform(self._line_scene(), width_px=900)
        mid_x, mid_y = t.to_canvas((5.0, 0.0))
        # tip is the first polygon vertex, at the arc-length midpoint
        assert xy[0, 0] == pytest.approx(mid_x, abs=0.51)
        assert xy[0, 1] == pytest.approx(mid_y, abs=0.51)
        # base vertices trail the tip: pointing +x
        assert xy[1:, 0].max() < xy[0, 0]

    def test_hollow_marker_is_outlined(self):
        svg = to_svg(self._line_scene(arrows=(0.5,), arrow_style="hollow"),
                     DEFAULT_STYLE, width_px=900)
        assert 'fill="white"' in svg and "arrowhead" in svg

    def test_closed_curve_marker_on_closing_segment(self):
        theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        circle = np.column_stack([np.cos(theta), np.sin(theta)])
        svg = to_svg(Scene(items=[Curve(circle, closed=True, arrows=(0.999,))]),
                     DEFAULT_STYLE, width_px=900)
        assert "arrowhead" in svg


class TestDashChannel:
    def test_semantic_dash_resolves(self):
        assert DEFAULT_STYLE.dash("dashed") == DEFAULT_STYLE.dash_patterns["dashed"]
        assert DEFAULT_STYLE.dash("solid") is None
        assert DEFAULT_STYLE.dash("2 7") == "2 7"

    def test_curve_dash_overrides_role(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0]])
        dashed = to_svg(Scene(items=[Curve(pts, dash="dashed")],
                              ylim=(-1, 1)), DEFAULT_STYLE)
        solid = to_svg(Scene(items=[Curve(pts, role=Role.CONSTRUCTION, dash="solid")],
                             ylim=(-1, 1)), DEFAULT_STYLE)
        assert "stroke-dasharray" in dashed
        assert "stroke-dasharray" not in solid  # solid strips the role's dash


class TestGhostVector:
    def test_open_head_uses_background_fill(self):
        svg = to_svg(Scene(items=[Vector((0.0, 0.0), (1.0, 0.0), filled=False)],
                           ylim=(-1, 1)), DEFAULT_STYLE)
        assert 'fill="white"' in svg and "arrowhead" in svg
