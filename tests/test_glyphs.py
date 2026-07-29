"""Annotation glyph pack + raster scalar field: brace, callout, rotated
labels, pattern fills, evenodd holes, scene clipping, RasterField."""

import base64
import io

import numpy as np
import pytest
from svgkit import svg_root, tag

from figlib.figure import Figure, Panel
from figlib.gates import _label_boxes, mechanical
from figlib.layout import Transform, geometry_extents
from figlib.render import brace_ink, callout_ink, figure_to_svg, to_svg, to_svg_tree
from figlib.scene import (Brace, Callout, Curve, FilledCurve, MathLabel,
                          RasterField, Scene)
from figlib.style import DEFAULT_STYLE, Role
from figlib.typeset import render_math


def box_scene(*extra, **scene_kw) -> Scene:
    """A 10x10 frame so extents are fixed and labels land well inside."""
    pts = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    kw = {"xlim": (0, 10), "ylim": (0, 10)}
    kw.update(scene_kw)
    return Scene(items=[Curve(pts, closed=True), *extra], **kw)


class TestBrace:
    def test_path_is_two_mirrored_halves(self):
        svg = to_svg(box_scene(Brace((2, 5), (8, 5))), DEFAULT_STYLE, width_px=900)
        root = svg_root(svg)
        braces = [e for e in root.iter() if e.get("class") == "brace"]
        assert len(braces) == 1
        d = braces[0].get("d")
        # hook C, shoulder L, cusp C — mirrored: 4 cubics, 2 lines
        assert d.startswith("M") and d.count("C") == 4 and d.count("L") == 2

    def test_cusp_centered_and_on_bulge_side(self):
        from figlib.render import BRACE_CUSP
        scene = box_scene(Brace((2, 5), (8, 5), side=1.0, depth=1.0))
        t = Transform(scene, width_px=900)
        cpts, _ = brace_ink(scene.items[1], t, DEFAULT_STYLE)
        mx, my = t.to_canvas((5.0, 6.0))     # midpoint + depth*left-normal (+y math)
        assert cpts[BRACE_CUSP][0] == pytest.approx(mx, abs=0.01)
        assert cpts[BRACE_CUSP][1] == pytest.approx(my, abs=0.01)
        # side=-1 mirrors the cusp across the span
        cpts2, _ = brace_ink(Brace((2, 5), (8, 5), side=-1.0, depth=1.0), t, DEFAULT_STYLE)
        _, my2 = t.to_canvas((5.0, 4.0))
        assert cpts2[BRACE_CUSP][1] == pytest.approx(my2, abs=0.01)
        # endpoints land exactly on p1 and p2
        e1, e2 = t.to_canvas((2.0, 5.0)), t.to_canvas((8.0, 5.0))
        assert cpts[0][0] == pytest.approx(e1[0]) and cpts[-1][0] == pytest.approx(e2[0])

    def test_default_depth_is_six_percent_of_span(self):
        _, _, n, d = Brace((0, 0), (10, 0)).frame()
        assert d == pytest.approx(0.6)
        assert n @ np.array([0.0, 1.0]) == pytest.approx(1.0)  # left of +x

    def test_label_box_collides_when_planted(self):
        brace = Brace((2, 5), (8, 5), depth=0.5, label=r"\delta")
        planted = MathLabel(r"x", (5.0, 5.7), ha="center", va="center")
        diags = mechanical(box_scene(brace, planted), DEFAULT_STYLE, width_px=900)
        assert any(d.kind == "label-collision" for d in diags)
        # without the planted label the brace label is clean
        assert mechanical(box_scene(brace), DEFAULT_STYLE, width_px=900) == []

    def test_extents_include_apex(self):
        (x0, x1), (y0, y1) = geometry_extents(
            Scene(items=[Brace((0.0, 0.0), (10.0, 0.0), depth=2.0)]))
        assert (x0, x1) == (0.0, 10.0)
        assert y1 == pytest.approx(2.0) and y0 == 0.0


class TestCallout:
    def _scene(self, **kw) -> Scene:
        return box_scene(Callout(r"x^2", anchor=(7.0, 7.0), target=(3.0, 3.0), **kw))

    def test_box_wraps_label_with_padding(self):
        scene = self._scene()
        t = Transform(scene, width_px=900)
        ink = callout_ink(scene.items[1], t, DEFAULT_STYLE)
        m = render_math(r"x^2", DEFAULT_STYLE.label_size_pt)
        x0, y0, x1, y1 = ink.box
        assert (x1 - x0) == pytest.approx(m.width_px + 10.0)
        assert (y1 - y0) == pytest.approx(m.height_px + 10.0)
        cx, cy = t.to_canvas((7.0, 7.0))
        assert (x0 + x1) / 2 == pytest.approx(cx) and (y0 + y1) / 2 == pytest.approx(cy)

    def test_leader_runs_from_nearest_box_edge_to_target(self):
        scene = self._scene()
        t = Transform(scene, width_px=900)
        ink = callout_ink(scene.items[1], t, DEFAULT_STYLE)
        tx, ty = t.to_canvas((3.0, 3.0))
        assert ink.leader[1][0] == pytest.approx(tx)
        assert ink.leader[1][1] == pytest.approx(ty)
        # start = clamp(target, box): on the boundary, nearest the target
        x0, y0, x1, y1 = ink.box
        sx, sy = ink.leader[0]
        assert sx == pytest.approx(min(max(tx, x0), x1))
        assert sy == pytest.approx(min(max(ty, y0), y1))
        assert sx in (pytest.approx(x0), pytest.approx(x1)) or \
               sy in (pytest.approx(y0), pytest.approx(y1))

    def test_svg_has_rounded_box_and_scaled_head(self):
        svg = to_svg(self._scene(), DEFAULT_STYLE, width_px=900)
        root = svg_root(svg)
        boxes = [e for e in root.iter() if e.get("class") == "callout-box"]
        assert len(boxes) == 1 and boxes[0].get("rx") == "6.00"
        heads = [e for e in root.iter() if e.get("class") == "arrowhead"]
        assert len(heads) == 1

    def test_box_participates_in_collisions_and_clipping(self):
        planted = MathLabel(r"y", (7.0, 7.0), ha="center", va="center")
        diags = mechanical(box_scene(self._scene().items[1], planted),
                           DEFAULT_STYLE, width_px=900)
        assert any(d.kind == "label-collision" for d in diags)
        # anchor beyond the fixed xlim/ylim: the box lands past the canvas edge
        off = box_scene(Callout(r"\int f", anchor=(11.0, 11.0), target=(5.0, 5.0)))
        assert any(d.kind == "clipped"
                   for d in mechanical(off, DEFAULT_STYLE, width_px=900))


class TestRotatedLabel:
    def test_rotated_bbox_is_aabb_of_rotated_rect(self):
        lab = lambda a: MathLabel(r"x + y + z", (5.0, 5.0), ha="center",
                                  va="center", angle_deg=a)
        style = DEFAULT_STYLE
        t = Transform(box_scene(), width_px=900)
        (_, _, flat), = _label_boxes(box_scene(lab(0.0)), style, t)
        (_, _, rot), = _label_boxes(box_scene(lab(30.0)), style, t)
        w, h = flat[2] - flat[0], flat[3] - flat[1]
        # wide label rotated 30deg: AABB height grows, width shrinks toward w*cos+h*sin
        assert (rot[3] - rot[1]) > h
        exp_w = w * np.cos(np.radians(30)) + h * np.sin(np.radians(30))
        exp_h = w * np.sin(np.radians(30)) + h * np.cos(np.radians(30))
        assert rot[2] - rot[0] == pytest.approx(exp_w, rel=1e-6)
        assert rot[3] - rot[1] == pytest.approx(exp_h, rel=1e-6)

    def test_render_emits_rotate_about_anchor(self):
        svg = to_svg(box_scene(MathLabel(r"x", (5.0, 5.0), angle_deg=30.0)),
                     DEFAULT_STYLE, width_px=900)
        assert "rotate(-30.00" in svg


class TestPatternFills:
    def _tri(self, off: float, pattern: str, color: str | None = None) -> FilledCurve:
        pts = np.array([[off, 0.0], [off + 2.0, 0.0], [off + 1.0, 2.0]])
        return FilledCurve(pts, pattern=pattern, color=color)

    def test_defs_emitted_once_per_pattern_color(self):
        scene = box_scene(self._tri(1, "stipple"), self._tri(4, "stipple"),
                          self._tri(7, "stipple"))
        root = svg_root(to_svg(scene, DEFAULT_STYLE, width_px=900))
        pats = [e for e in root.iter() if tag(e) == "pattern"
                and e.get("id", "").startswith("pat-")]
        assert len(pats) == 1
        fills = [e.get("fill") for e in root.iter()
                 if (e.get("fill") or "").startswith("url(#pat-")]
        assert len(fills) == 3 and len(set(fills)) == 1

    def test_distinct_colors_get_distinct_defs(self):
        scene = box_scene(self._tri(1, "hatch", "#112233"),
                          self._tri(4, "hatch", "#445566"),
                          self._tri(7, "crosshatch", "#112233"))
        root = svg_root(to_svg(scene, DEFAULT_STYLE, width_px=900))
        ids = {e.get("id") for e in root.iter() if tag(e) == "pattern"
               and e.get("id", "").startswith("pat-")}
        assert ids == {"pat-hatch-112233", "pat-hatch-445566",
                       "pat-crosshatch-112233"}

    def test_pattern_tile_ground_is_transparent(self):
        root = svg_root(to_svg(box_scene(self._tri(1, "stipple")),
                               DEFAULT_STYLE, width_px=900))
        pat = next(e for e in root.iter() if tag(e) == "pattern"
                   and e.get("id", "").startswith("pat-"))
        assert all(tag(child) in ("circle", "line") for child in pat)  # no bg rect


class TestHoles:
    def test_evenodd_multi_subpath(self):
        theta = np.linspace(0, 2 * np.pi, 60, endpoint=False)
        ring = np.column_stack([np.cos(theta), np.sin(theta)])
        fc = FilledCurve(2 * ring + 5.0, holes=(ring + 5.0,))
        root = svg_root(to_svg(box_scene(fc), DEFAULT_STYLE, width_px=900))
        path = next(e for e in root.iter()
                    if tag(e) == "path" and e.get("fill-rule") == "evenodd")
        assert path.get("d").count("Z") == 2
        assert path.get("d").count("M") == 2


class TestClip:
    def test_frame_clip_single_scene(self):
        scene = box_scene(clip="frame")
        root, t = to_svg_tree(scene, DEFAULT_STYLE, width_px=900)
        cps = [e for e in root.iter() if e.tag == "clipPath"]
        assert len(cps) == 1
        rect = cps[0].find("rect")
        x0, y0 = t.to_canvas((0.0, 10.0))
        assert float(rect.get("x")) == pytest.approx(x0, abs=0.01)
        assert float(rect.get("y")) == pytest.approx(y0, abs=0.01)
        gs = [e for e in root.iter() if e.tag == "g" and e.get("clip-path")]
        assert len(gs) == 1 and gs[0].get("clip-path") == f"url(#{cps[0].get('id')})"

    def test_polygon_clip(self):
        disc = np.column_stack([np.cos(np.linspace(0, 2 * np.pi, 40)),
                                np.sin(np.linspace(0, 2 * np.pi, 40))]) * 4 + 5
        root, _ = to_svg_tree(box_scene(clip_pts=disc), DEFAULT_STYLE, width_px=900)
        cp = next(e for e in root.iter() if e.tag == "clipPath")
        assert cp.find("path") is not None

    def test_clip_in_figure_panel(self):
        fig = Figure(panels=[Panel(box_scene(clip="frame"))])
        root = svg_root(figure_to_svg(fig, DEFAULT_STYLE, width_px=900))
        assert any(tag(e) == "clipPath" for e in root.iter())
        # the clipped group lives inside the translated panel group
        panel_g = next(e for e in root.iter()
                       if tag(e) == "g" and (e.get("transform") or "").startswith("translate"))
        assert any(ch.get("clip-path") for ch in panel_g)

    def test_no_clip_no_clippath(self):
        root, _ = to_svg_tree(box_scene(), DEFAULT_STYLE, width_px=900)
        assert not any(e.tag == "clipPath" for e in root.iter())


class TestRasterField:
    def _field(self, **kw) -> RasterField:
        return RasterField(np.array([[0.0, 1.0], [2.0, 4.0]]),
                           extent=(2.0, 8.0, 3.0, 7.0), **kw)

    def _image(self, scene):
        root, t = to_svg_tree(scene, DEFAULT_STYLE, width_px=900)
        img = next(e for e in root.iter()
                   if e.tag == "image" and e.get("preserveAspectRatio") == "none")
        return img, t

    def test_extent_positioning_and_yflip(self):
        img, t = self._image(box_scene(self._field()))
        x, y = t.to_canvas((2.0, 7.0))          # row 0 = top = y1
        assert float(img.get("x")) == pytest.approx(x, abs=0.01)
        assert float(img.get("y")) == pytest.approx(y, abs=0.01)
        assert float(img.get("width")) == pytest.approx(6 * t.scale_x, abs=0.01)
        assert float(img.get("height")) == pytest.approx(4 * t.scale_y, abs=0.01)
        assert img.get("image-rendering") == "pixelated"

    def test_normalization_and_ramp(self):
        from PIL import Image
        ramp = lambda u: f"#{round(255 * u):02x}0000"      # red channel = u
        img, _ = self._image(box_scene(self._field(ramp=ramp, vmin=0.0, vmax=4.0)))
        data = base64.b64decode(img.get("href").split(",", 1)[1])
        px = np.asarray(Image.open(io.BytesIO(data)).convert("RGBA"))
        # pixelated -> nearest-upsampled by an integer factor; cells stay flat
        k = px.shape[0] // 2
        assert px.shape == (2 * k, 2 * k, 4) and k >= 1
        assert px[0, 0, 0] == 0                       # value 0 -> t=0
        assert px[-1, -1, 0] == 255                   # value 4 -> t=1
        assert px[0, -1, 0] == round(255 * 1 / 4)     # value 1, top-right
        assert np.all(px[:k, :k, 0] == 0)             # flat within a cell
        assert px[0, 0, 3] == 255                     # opaque alpha channel

    def test_vmin_vmax_clamp(self):
        from PIL import Image
        ramp = lambda u: f"#{round(255 * u):02x}0000"
        fld = RasterField(np.array([[-10.0, 10.0]]), extent=(0, 2, 0, 1),
                          ramp=ramp, vmin=0.0, vmax=1.0)
        img, _ = self._image(box_scene(fld))
        data = base64.b64decode(img.get("href").split(",", 1)[1])
        px = np.asarray(Image.open(io.BytesIO(data)).convert("RGBA"))
        assert px[0, 0, 0] == 0 and px[0, -1, 0] == 255

    def test_participates_in_extents(self):
        (x0, x1), (y0, y1) = geometry_extents(Scene(items=[self._field()]))
        assert (x0, x1) == (2.0, 8.0) and (y0, y1) == (3.0, 7.0)

    def test_opacity_and_interp_flags(self):
        img, _ = self._image(box_scene(self._field(opacity=0.5, interp=True)))
        assert img.get("opacity") == "0.50"
        assert img.get("image-rendering") is None
