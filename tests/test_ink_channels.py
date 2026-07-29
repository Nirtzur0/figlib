"""Continuous ink channels: width profiles, ramps along curves, casing,
clipping, raster fields — the attributes that vary along or under a stroke."""

import re
from xml.etree import ElementTree as ET

import numpy as np

from figlib.render import to_svg
from figlib.scene import Curve, Scene
from figlib.style import DEFAULT_STYLE, Role

NS = {"s": "http://www.w3.org/2000/svg"}


def _paths(svg: str) -> list[ET.Element]:
    return ET.fromstring(svg).findall(".//s:path", NS)


def _d_points(d: str) -> np.ndarray:
    nums = [float(v) for v in re.findall(r"-?\d+\.?\d*", d)]
    return np.array(nums).reshape(-1, 2)


class TestWidthProfile:
    def _svg(self) -> str:
        pts = np.column_stack([np.linspace(0.0, 1.0, 50), np.zeros(50)])
        sc = Scene(items=[Curve(pts, width_profile=(0.0, 1.0), width_scale=4.0)],
                   xlim=(0.0, 1.0), ylim=(-0.5, 0.5))
        return to_svg(sc, DEFAULT_STYLE, width_px=400)

    def test_emits_filled_polygon_not_stroke(self):
        polys = [p for p in _paths(self._svg()) if p.get("class") == "varstroke"]
        assert len(polys) == 1
        p = polys[0]
        assert p.get("fill") == DEFAULT_STYLE.ink(Role.CONTENT).color
        assert p.get("stroke") in (None, "none")

    def test_width_tapers_from_zero_to_full(self):
        polys = [p for p in _paths(self._svg()) if p.get("class") == "varstroke"]
        pts = _d_points(polys[0].get("d"))
        full_px = DEFAULT_STYLE.ink(Role.CONTENT).width * 4.0
        x_lo, x_hi = pts[:, 0].min(), pts[:, 0].max()
        near_start = pts[np.abs(pts[:, 0] - x_lo) < 2.0]
        near_end = pts[np.abs(pts[:, 0] - x_hi) < 2.0]
        spread_start = near_start[:, 1].max() - near_start[:, 1].min()
        spread_end = near_end[:, 1].max() - near_end[:, 1].min()
        assert spread_start < 0.5
        assert abs(spread_end - full_px) < 0.6

    def test_direction_markers_survive_on_profiled_curve(self):
        pts = np.column_stack([np.linspace(0.0, 1.0, 50), np.zeros(50)])
        sc = Scene(items=[Curve(pts, width_profile=(0.2, 1.0), arrows=(0.5,))],
                   xlim=(0.0, 1.0), ylim=(-0.5, 0.5))
        svg = to_svg(sc, DEFAULT_STYLE, width_px=400)
        heads = [p for p in ET.fromstring(svg).findall(".//s:polygon", NS)
                 if p.get("class") == "arrowhead"]
        assert len(heads) == 1


class TestRampSegments:
    def _pts(self) -> np.ndarray:
        th = np.linspace(0.0, 2.0 * np.pi, 200)
        return np.column_stack([th * np.cos(th), th * np.sin(th)])

    def test_partitions_curve_with_ramped_attributes(self):
        from figlib.builders import ramp_segments
        segs = ramp_segments(self._pts(), n=8,
                             opacity=lambda t: 0.1 + 0.9 * t,
                             color=lambda t: f"#0000{int(t * 255):02x}")
        assert len(segs) == 8
        assert all(isinstance(s, Curve) for s in segs)
        assert segs[0].opacity < segs[-1].opacity
        assert segs[0].color != segs[-1].color
        for a, b in zip(segs, segs[1:]):
            np.testing.assert_allclose(a.pts[-1], b.pts[0])
        # butt caps so translucent joints don't double-draw
        assert all(s.cap == "butt" for s in segs)

    def test_segments_cover_equal_arc_length(self):
        from figlib.builders import ramp_segments
        segs = ramp_segments(self._pts(), n=6)
        lengths = []
        for s in segs:
            d = np.diff(s.pts, axis=0)
            lengths.append(np.hypot(d[:, 0], d[:, 1]).sum())
        assert max(lengths) / min(lengths) < 1.05

    def test_cap_field_reaches_svg(self):
        pts = np.column_stack([np.linspace(0.0, 1.0, 10), np.zeros(10)])
        sc = Scene(items=[Curve(pts, cap="butt")],
                   xlim=(0.0, 1.0), ylim=(-0.5, 0.5))
        svg = to_svg(sc, DEFAULT_STYLE, width_px=400)
        caps = [p.get("stroke-linecap") for p in _paths(svg)]
        assert "butt" in caps


class TestCasing:
    def test_label_halo_emits_paper_stroked_copy_beneath_ink(self):
        from figlib.scene import MathLabel
        sc = Scene(items=[MathLabel(r"x", (0.5, 0.0), halo=True)],
                   xlim=(0.0, 1.0), ylim=(-0.5, 0.5))
        svg = to_svg(sc, DEFAULT_STYLE, width_px=400)
        els = list(ET.fromstring(svg).iter())
        halo = [i for i, e in enumerate(els)
                if e.get("stroke") == DEFAULT_STYLE.background
                and e.get("fill") == DEFAULT_STYLE.background]
        ink = [i for i, e in enumerate(els)
               if e.get("fill") == DEFAULT_STYLE.ink(Role.CONTENT).color]
        assert halo, "no paper-stroked glyph copy found"
        assert ink, "no ink glyph copy found"
        assert max(halo) < min(ink), "halo must render beneath the ink copy"
        # never mutate shared <symbol> defs — the stroke would leak into
        # every glyph copy, including the ink one
        for sym in ET.fromstring(svg).iter():
            if sym.tag.split("}")[-1] == "symbol":
                assert all("stroke" not in el.attrib for el in sym.iter())

    def test_curve_casing_emits_wider_paper_stroke_first(self):
        pts = np.column_stack([np.linspace(0.0, 1.0, 10), np.zeros(10)])
        sc = Scene(items=[Curve(pts, casing=True)],
                   xlim=(0.0, 1.0), ylim=(-0.5, 0.5))
        svg = to_svg(sc, DEFAULT_STYLE, width_px=400)
        paths = _paths(svg)
        base_w = DEFAULT_STYLE.ink(Role.CONTENT).width
        casing = [i for i, p in enumerate(paths)
                  if p.get("stroke") == DEFAULT_STYLE.background
                  and float(p.get("stroke-width", 0)) > base_w]
        ink = [i for i, p in enumerate(paths)
               if p.get("stroke") == DEFAULT_STYLE.ink(Role.CONTENT).color]
        assert len(casing) == 1
        assert casing[0] < ink[0], "casing must render beneath its own stroke"

    def test_halo_skipped_when_transparent(self):
        from dataclasses import replace
        from figlib.scene import MathLabel
        from figlib.theme import CLEAN, transparent_variant
        sc = Scene(items=[MathLabel(r"x", (0.5, 0.0), halo=True)],
                   xlim=(0.0, 1.0), ylim=(-0.5, 0.5))
        svg = to_svg(sc, transparent_variant(CLEAN), width_px=400)
        stroked = [e for e in ET.fromstring(svg).iter()
                   if e.get("stroke") == CLEAN.background and e.get("fill") == CLEAN.background]
        assert not stroked
