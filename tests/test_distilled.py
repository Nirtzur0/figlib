"""Distilled builder judgment, now deterministic: ink corridors and
arrow-on-mark gates, auto level ladders, tangent alignment, PNG zoom,
compliant-ink repair, hidden variants, fill-as-ground contrast, cross2d."""

import numpy as np
import pytest
from svgkit import find_by, svg_root

from figlib.autoplace import autoplace
from figlib.builders import (assert_tangents_align, auto_levels,
                             stream_function_lines)
from figlib.color import (compliant, composite, contrast,
                          min_compliant_opacity, to_oklab)
from figlib.gates import (_SCAFFOLD_ROLES, MIN_PERCEPTIBLE_CONTRAST,
                          MIN_STROKE_CONTRAST, color_gate, mechanical)
from figlib.geometry import cross2d, point_in_poly
from figlib.layout import Transform
from figlib.render import PNG_SCALE, to_svg
from figlib.scene import (Curve, FilledCurve, MathLabel, Point, Scene,
                          Vector)
from figlib.style import DEFAULT_STYLE, Role
from figlib.theme import CLEAN, RISO


def _kinds(diags):
    return {d.kind for d in diags}


def _diagonal_scene(*extra, role=Role.CONTENT):
    return Scene(items=[Curve(np.array([[0.0, 0.0], [10.0, 10.0]]), role=role),
                        *extra],
                 xlim=(0, 10), ylim=(0, 10))


class TestLabelOnInk:
    def _label(self, **kw):
        kw.setdefault("ha", "center")
        kw.setdefault("va", "center")
        return MathLabel(r"x", (5.0, 5.0), **kw)

    def test_label_on_content_curve_flagged_with_free_nudge(self):
        diags = mechanical(_diagonal_scene(self._label(pin=True)),
                           DEFAULT_STYLE)
        hits = [d for d in diags if d.kind == "label-on-ink"]
        assert len(hits) == 1
        assert "Curve CONTENT" in hits[0].detail
        assert "offset_px" in hits[0].detail

    def test_scaffold_roles_exempt(self):
        for role in (Role.CONSTRUCTION, Role.FRAME, Role.MUTED):
            diags = mechanical(_diagonal_scene(self._label(), role=role),
                               DEFAULT_STYLE)
            assert "label-on-ink" not in _kinds(diags)

    def test_vector_ink_is_a_corridor_too(self):
        scene = Scene(items=[Vector((0.0, 5.0), (10.0, 5.0)),
                             self._label(pin=True)],
                      xlim=(0, 10), ylim=(0, 10))
        diags = mechanical(scene, DEFAULT_STYLE)
        assert any(d.kind == "label-on-ink" and "Vector" in d.detail
                   for d in diags)

    def test_autoplace_moves_label_off_ink(self):
        lab = self._label()
        scene = _diagonal_scene(lab)
        notes = autoplace(scene, DEFAULT_STYLE)
        assert notes and lab.offset_px != (0.0, 0.0)
        assert "label-on-ink" not in _kinds(mechanical(scene, DEFAULT_STYLE))

    def test_pinned_label_stays_and_gate_reports(self):
        lab = self._label(pin=True)
        scene = _diagonal_scene(lab)
        autoplace(scene, DEFAULT_STYLE)
        assert lab.offset_px == (0.0, 0.0)
        assert "label-on-ink" in _kinds(mechanical(scene, DEFAULT_STYLE))

    def test_halo_inflates_the_checked_box(self):
        # anchored just off the stroke: bare glyphs clear, the halo cuts it
        off = 6.5 / Transform(_diagonal_scene(), width_px=900).scale
        bare = MathLabel(r"x", (5.0 + off, 5.0 - off), ha="center", va="center")
        scene = _diagonal_scene(bare)
        assert "label-on-ink" not in _kinds(mechanical(scene, DEFAULT_STYLE))
        haloed = MathLabel(r"x", (5.0 + off, 5.0 - off), ha="center",
                           va="center", halo=True, pin=True)
        diags = mechanical(_diagonal_scene(haloed), DEFAULT_STYLE)
        hits = [d for d in diags if d.kind == "label-on-ink"]
        assert hits and "halo" in hits[0].detail

    def test_no_free_nudge_reports_ink_free_region_in_math_coords(self):
        # a dense block of content lines: no single-axis escape within
        # budget, so the diagnostic names the nearest free region center
        lines = [Curve(np.array([[0.0, y], [10.0, y]]))
                 for y in np.arange(0.0, 6.05, 0.1)]
        lab = MathLabel(r"x", (5.0, 3.0), ha="center", va="center", pin=True)
        scene = Scene(items=[*lines, lab], xlim=(0, 10), ylim=(0, 10))
        diags = mechanical(scene, DEFAULT_STYLE, width_px=450)
        hits = [d for d in diags if d.kind == "label-on-ink"]
        assert hits and "math coords" in hits[0].detail
        assert "ink-free region" in hits[0].detail


class TestArrowOnMark:
    def _flow(self, arrows=(0.5,), role=Role.CONTENT, extra=()):
        return Scene(items=[
            Curve(np.array([[0.0, 5.0], [10.0, 5.0]]), role=role,
                  arrows=arrows), *extra],
            xlim=(0, 10), ylim=(0, 10))

    def test_marker_on_point_flagged_with_clear_fraction(self):
        scene = self._flow(extra=[Point((5.0, 5.0))])
        diags = mechanical(scene, DEFAULT_STYLE)
        hits = [d for d in diags if d.kind == "arrow-on-mark"]
        assert len(hits) == 1
        assert "Point" in hits[0].detail and "clear: arrows=(" in hits[0].detail

    def test_marker_on_label_flagged(self):
        lab = MathLabel(r"\Psi", (5.0, 5.0), ha="center", va="center", pin=True)
        diags = mechanical(self._flow(extra=[lab]), DEFAULT_STYLE)
        assert any(d.kind == "arrow-on-mark" and r"\Psi" in d.detail
                   for d in diags)

    def test_annotation_axis_arrow_exempt(self):
        # an axis arrow ends AT its label by design
        lab = MathLabel(r"t", (10.0, 5.0), ha="left", va="center", pin=True)
        scene = self._flow(arrows=(1.0,), role=Role.ANNOTATION, extra=[lab])
        assert "arrow-on-mark" not in _kinds(mechanical(scene, DEFAULT_STYLE))

    def test_clear_marker_not_flagged(self):
        scene = self._flow(extra=[Point((9.0, 5.0))])
        assert "arrow-on-mark" not in _kinds(mechanical(scene, DEFAULT_STYLE))


class TestAutoLevels:
    def test_linear_field_snaps_to_unit_ladder(self):
        levels = auto_levels(lambda X, Y: X, (0, 10), (0, 10))
        assert levels == pytest.approx(list(range(1, 10)))

    def test_one_two_five_ladder(self):
        levels = auto_levels(lambda X, Y: X, (0, 26), (0, 1))
        steps = np.diff(levels)
        assert np.allclose(steps, 2.0)
        assert all(abs(lv / 2.0 - round(lv / 2.0)) < 1e-9 for lv in levels)

    def test_exclude_drops_the_separatrix_level(self):
        levels = auto_levels(lambda X, Y: X, (0, 10), (0, 10), exclude=(5.0,))
        assert 5.0 not in levels and 4.0 in levels and 6.0 in levels

    def test_mask_restricts_the_range(self):
        levels = auto_levels(lambda X, Y: X, (0, 10), (0, 10),
                             mask=lambda X, Y: X > 5.0)
        assert levels and max(levels) < 5.0

    def test_percentiles_ignore_spikes(self):
        def psi(X, Y):
            Z = np.asarray(X, dtype=float).copy()
            Z[np.hypot(X - 5.0, Y - 5.0) < 0.2] = 1e6   # a pole
            return Z
        levels = auto_levels(psi, (0, 10), (0, 10))
        assert levels and max(levels) < 20.0


class TestStubCullAndMarkerSuppression:
    PSI = staticmethod(lambda X, Y: X**2 + Y**2)
    DOM = ((-2.0, 2.0), (-2.0, 2.0))                    # diagonal ~5.66

    def _circle(self, r, **kw):
        return stream_function_lines(self.PSI, *self.DOM, levels=[r * r],
                                     n=401, **kw)

    def test_tiny_stub_dropped(self):
        assert self._circle(0.02) == []

    def test_min_arc_frac_zero_keeps_it(self):
        assert len(self._circle(0.02, min_arc_frac=0.0)) == 1

    def test_short_line_keeps_ink_but_loses_marker(self):
        curves = self._circle(0.04)
        assert len(curves) == 1 and curves[0].arrows == ()

    def test_long_line_keeps_marker(self):
        curves = self._circle(1.0)
        assert len(curves) == 1 and curves[0].arrows == (0.5,)

    def test_min_marker_arc_override(self):
        curves = self._circle(1.0, min_marker_arc=100.0)
        assert curves[0].arrows == ()


class TestAssertTangentsAlign:
    def test_aligned_flow_passes(self):
        curves = stream_function_lines(lambda X, Y: Y, (-2, 2), (-1, 1),
                                       levels=[-0.5, 0.0, 0.5], n=101)
        field = lambda P: np.column_stack([np.ones(len(P)), np.zeros(len(P))])
        assert_tangents_align(curves, field)

    def test_reversed_field_fails_signed(self):
        curves = stream_function_lines(lambda X, Y: Y, (-2, 2), (-1, 1),
                                       levels=[0.0], n=101)
        field = lambda P: np.column_stack([-np.ones(len(P)), np.zeros(len(P))])
        with pytest.raises(AssertionError, match="deg"):
            assert_tangents_align(curves, field)

    def test_exclude_near_skips_singular_samples(self):
        curves = [Curve(np.column_stack([np.linspace(-2, 2, 200),
                                         np.zeros(200)]))]

        def field(P):
            v = np.column_stack([np.ones(len(P)), np.zeros(len(P))])
            near = np.hypot(P[:, 0], P[:, 1]) < 0.3
            v[near] = (0.0, 1.0)                        # field turns near 0
            return v

        with pytest.raises(AssertionError):
            assert_tangents_align(curves, field)
        assert_tangents_align(curves, field, exclude_near=[(0.0, 0.0)],
                              exclude_radius=0.6)


ZOOM_PROGRAM = '''
import numpy as np
from figlib.scene import Curve, Scene

CLAIM = "A box."
PARAMS = {}

def compute(p):
    return np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])

def build(g):
    return Scene(items=[Curve(g, closed=True)], xlim=(0, 10), ylim=(0, 10))

def assertions(g):
    pass
'''


class TestFigcheckZoom:
    def test_zoom_crop_is_math_window_at_png_scale(self, tmp_path):
        from figlib.cli import _zoom_main
        from figlib.program import run
        from PIL import Image

        prog = tmp_path / "toy_zoom.py"
        prog.write_text(ZOOM_PROGRAM)
        report = run(prog, out_dir=tmp_path / "out")
        _zoom_main(report, "2,2,8,8:2")
        out = report.png_path.with_name(report.png_path.stem + ".zoom.png")
        assert out.exists()
        t = Transform(report.built, width_px=report.width_px)
        (cx0, cy0), (cx1, cy1) = t.to_canvas((2.0, 8.0)), t.to_canvas((8.0, 2.0))
        w = int(cx1 * PNG_SCALE + 0.5) - int(cx0 * PNG_SCALE)
        h = int(cy1 * PNG_SCALE + 0.5) - int(cy0 * PNG_SCALE)
        with Image.open(out) as im:
            assert im.size == (2 * w, 2 * h)

    def test_cli_flag_prints_the_path(self, tmp_path, capsys):
        from figlib.cli import main

        prog = tmp_path / "toy_zoom2.py"
        prog.write_text(ZOOM_PROGRAM)
        assert main([str(prog), "--zoom", "1,1,9,9"]) == 0
        assert "zoom:" in capsys.readouterr().out


class TestCompliant:
    def test_reaches_the_floor_preserving_hue(self):
        fixed = compliant("#c6d2e4", "#ffffff", 3.0)
        assert contrast(fixed, "#ffffff") >= 3.0
        _, a0, b0 = to_oklab("#c6d2e4")
        _, a1, b1 = to_oklab(fixed)
        h0, h1 = np.arctan2(b0, a0), np.arctan2(b1, a1)
        assert abs(np.degrees(np.arctan2(np.sin(h1 - h0),
                                         np.cos(h1 - h0)))) < 15.0

    def test_already_compliant_returns_the_ink(self):
        assert compliant("#1a1a1a", "#ffffff", 3.0) == "#1a1a1a"

    def test_opacity_is_measured_on_the_composite(self):
        fixed = compliant("#5f6f8f", "#ffffff", 3.0, opacity=0.5)
        assert fixed is not None
        assert contrast(composite(fixed, 0.5, "#ffffff"), "#ffffff") >= 3.0

    def test_none_when_no_lightness_clears(self):
        # at opacity 0.1 even pure black composites to a pale gray
        assert compliant("#1a1a1a", "#ffffff", 3.0, opacity=0.1) is None

    def test_min_compliant_opacity(self):
        alpha = min_compliant_opacity("#1a1a1a", "#ffffff", 3.0)
        assert alpha is not None and alpha < 1.0
        assert contrast(composite("#1a1a1a", alpha, "#ffffff"), "#ffffff") >= 3.0

    def test_faint_ink_diagnostic_carries_the_repair(self):
        scene = Scene(items=[Curve(np.array([[0.0, 0.0], [1.0, 1.0]]),
                                   color="#dddddd")],
                      xlim=(0, 1), ylim=(0, 1))
        diags = color_gate(scene, DEFAULT_STYLE)
        hits = [d for d in diags if d.kind == "faint-ink"]
        assert hits and "nearest compliant: #" in hits[0].detail


class TestStyleParity:
    def test_fine_dashed_named_pattern(self):
        assert DEFAULT_STYLE.dash("fine-dashed") == "4 3.5"

    def test_raw_dash_scales_with_type_scale(self):
        s = DEFAULT_STYLE.scaled(2.0)
        assert s.dash("4 2") == "8 4"
        assert s.dash("dashed") == "12 9"       # named patterns still scale

    def test_raw_dash_untouched_at_unit_scale(self):
        assert DEFAULT_STYLE.dash("4 2") == "4 2"

    def test_point_color_override_renders(self):
        scene = Scene(items=[Point((0.5, 0.5), color="#123456"),
                             Point((0.2, 0.2), filled=False, color="#654321")],
                      xlim=(0, 1), ylim=(0, 1))
        root = svg_root(to_svg(scene, DEFAULT_STYLE, width_px=300))
        assert find_by(root, "circle", fill="#123456")
        assert find_by(root, "circle", stroke="#654321")

    def test_hidden_variant_drives_sphere_and_wireframe(self):
        from figlib.sphere3d import Sphere, circle_on_sphere
        from figlib.surface3d import Camera, wireframe_items

        _, o_mult, w_mult = DEFAULT_STYLE.hidden_variant(Role.CONTENT)
        cam = Camera()
        items = circle_on_sphere(Sphere(np.zeros(3), 1.0), (0, 0, 1), 0.0, cam)
        hidden = [c for _, c in items if c.dash is not None]
        assert hidden
        assert all(c.opacity == pytest.approx(o_mult) for c in hidden)
        assert all(c.width_scale == pytest.approx(w_mult) for c in hidden)

        u = np.linspace(-1, 1, 9)
        X, Y = np.meshgrid(u, u)
        Z = X**2 - Y**2
        witems = wireframe_items(X, Y, Z, cam, stride=(4, 4), hidden="dashed")
        whidden = [c for _, c in witems if c.dash is not None]
        assert whidden
        assert all(c.opacity == pytest.approx(o_mult) for c in whidden)

    @pytest.mark.parametrize("theme", [CLEAN, RISO], ids=["clean", "riso"])
    def test_every_hidden_variant_clears_its_contrast_floor(self, theme):
        for role in Role:
            _, o_mult, _ = theme.hidden_variant(role)
            ink = theme.ink(role)
            worst = min(contrast(composite(ink.color, o_mult, p), p)
                        for p in theme.paper_stops())
            floor = (MIN_PERCEPTIBLE_CONTRAST if role in _SCAFFOLD_ROLES
                     else MIN_STROKE_CONTRAST)
            assert worst >= floor, (
                f"{role.name} hidden variant at {worst:.2f}:1 < {floor}")


# a mark that clears 3:1 on white but not on a near-black fill
_MID_GRAY = "#626262"
_DARK = "#33302e"


class TestContrastVsFill:
    def _square(self, lo=0.0, hi=10.0, **kw):
        pts = np.array([[lo, lo], [hi, lo], [hi, hi], [lo, hi]])
        return FilledCurve(pts, outline=False, **kw)

    def test_mark_inside_high_opacity_fill_measured_against_it(self):
        scene = Scene(items=[self._square(color=_DARK, opacity=1.0),
                             Point((5.0, 5.0), color=_MID_GRAY)],
                      xlim=(0, 10), ylim=(0, 10))
        hits = [d for d in color_gate(scene, DEFAULT_STYLE)
                if d.kind == "faint-ink"]
        assert hits and "on fill" in hits[0].detail

    def test_same_mark_on_paper_passes(self):
        scene = Scene(items=[Point((5.0, 5.0), color=_MID_GRAY)],
                      xlim=(0, 10), ylim=(0, 10))
        assert color_gate(scene, DEFAULT_STYLE) == []

    def test_low_opacity_fill_is_not_a_ground(self):
        scene = Scene(items=[self._square(color=_DARK, opacity=0.3),
                             Point((5.0, 5.0), color=_MID_GRAY)],
                      xlim=(0, 10), ylim=(0, 10))
        assert "faint-ink" not in _kinds(color_gate(scene, DEFAULT_STYLE))

    def test_topmost_containing_fill_wins(self):
        scene = Scene(items=[self._square(color=_DARK, opacity=1.0),
                             self._square(3.0, 7.0, color="#d8d8d8",
                                          opacity=1.0),
                             Point((5.0, 5.0), color=_MID_GRAY)],
                      xlim=(0, 10), ylim=(0, 10))
        assert "faint-ink" not in _kinds(color_gate(scene, DEFAULT_STYLE))

    def test_label_offset_moves_it_off_the_fill(self):
        # anchored inside the fill, offset onto the paper: no diagnostic
        lab = MathLabel(r"x^*", (5.0, 5.0), color=_MID_GRAY,
                        offset_px=(400.0, 0.0))
        scene = Scene(items=[self._square(hi=5.5, color=_DARK, opacity=1.0),
                             lab],
                      xlim=(0, 10), ylim=(0, 10))
        assert "faint-ink" not in _kinds(color_gate(scene, DEFAULT_STYLE))


class TestCross2d:
    def test_scalar_pairs(self):
        assert cross2d((1.0, 0.0), (0.0, 1.0)) == 1.0
        assert cross2d((0.0, 1.0), (1.0, 0.0)) == -1.0

    def test_broadcasts_and_matches_formula(self):
        rng = np.random.default_rng(3)
        a = rng.standard_normal((5, 7, 2))
        b = rng.standard_normal((5, 7, 2))
        expect = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
        assert np.allclose(cross2d(a, b), expect)
        assert np.allclose(cross2d(a, np.array([1.0, 2.0])),
                           2.0 * a[..., 0] - a[..., 1])

    def test_point_in_poly(self):
        sq = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        assert point_in_poly((0.5, 0.5), sq)
        assert not point_in_poly((1.5, 0.5), sq)
        assert not point_in_poly((-0.1, 0.99), sq)
