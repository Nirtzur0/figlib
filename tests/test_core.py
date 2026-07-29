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
