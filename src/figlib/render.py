"""SVG emission and PNG rasterization.

Arrowheads are explicit filled polygons (not markers) so every piece of
ink has a bbox we own.
"""

from __future__ import annotations

import math
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from .layout import Transform
from .scene import (AngleMark, Curve, FilledCurve, MathLabel, Point,
                    RightAngleMark, Scene, Vector)
from .style import DEFAULT_STYLE, Role, Style
from .typeset import draw_math

SVG_NS = "http://www.w3.org/2000/svg"


def _fmt(v: float) -> str:
    return f"{v:.2f}"


def _path_d(pts: np.ndarray, closed: bool) -> str:
    parts = [f"M {_fmt(pts[0, 0])} {_fmt(pts[0, 1])}"]
    parts += [f"L {_fmt(x)} {_fmt(y)}" for x, y in pts[1:]]
    if closed:
        parts.append("Z")
    return " ".join(parts)


def _add_stroke(el: ET.Element, style: Style, role: Role, width_scale: float = 1.0) -> None:
    ink = style.ink(role)
    el.set("stroke", ink.color)
    el.set("stroke-width", _fmt(ink.width * width_scale))
    el.set("stroke-linecap", "round")
    el.set("stroke-linejoin", "round")
    if ink.dash:
        el.set("stroke-dasharray", ink.dash)


def _arrowhead(tip: tuple[float, float], direction: tuple[float, float], style: Style,
               scale: float = 1.0) -> list[tuple[float, float]]:
    dx, dy = direction
    n = math.hypot(dx, dy) or 1.0
    ux, uy = dx / n, dy / n
    px, py = -uy, ux
    L, W = style.arrowhead_len * scale, style.arrowhead_halfwidth * scale
    bx, by = tip[0] - L * ux, tip[1] - L * uy
    return [tip, (bx + W * px, by + W * py), (bx - W * px, by - W * py)]


def _emit_head(root: ET.Element, head: list[tuple[float, float]], color: str,
               style: Style, hollow: bool, stroke_width: float) -> None:
    attrs = {"class": "arrowhead",
             "points": " ".join(f"{_fmt(x)},{_fmt(y)}" for x, y in head)}
    if hollow:
        transparent = getattr(style, "transparent", False)
        attrs.update({"fill": "none" if transparent else style.background,
                      "stroke": color, "stroke-width": _fmt(stroke_width),
                      "stroke-linejoin": "round"})
    else:
        attrs["fill"] = color
    ET.SubElement(root, "polygon", attrs)


def _at_fraction(pts: np.ndarray, t: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """Point and tangent at arc-length fraction t of a canvas polyline."""
    seg = np.diff(pts, axis=0)
    lens = np.hypot(seg[:, 0], seg[:, 1])
    total = float(lens.sum())
    if total == 0.0:
        return (float(pts[0, 0]), float(pts[0, 1])), (1.0, 0.0)
    target = min(max(t, 0.0), 1.0) * total
    cum = np.cumsum(lens)
    i = min(int(np.searchsorted(cum, target)), len(lens) - 1)
    prev = float(cum[i - 1]) if i > 0 else 0.0
    f = (target - prev) / (lens[i] or 1.0)
    p = pts[i] + f * seg[i]
    # tangent from the surrounding segment; skip zero-length segments
    d = seg[i]
    if lens[i] == 0.0:
        j = next((k for k in range(len(lens)) if lens[k] > 0), None)
        d = seg[j] if j is not None else np.array([1.0, 0.0])
    return (float(p[0]), float(p[1])), (float(d[0]), float(d[1]))


def to_svg_tree(scene: Scene, style: Style = DEFAULT_STYLE, width_px: float = 900) -> tuple[ET.Element, Transform]:
    t = Transform(scene, width_px=width_px)
    root = ET.Element(
        "svg",
        {
            "xmlns": SVG_NS,
            "width": _fmt(t.canvas_w),
            "height": _fmt(t.canvas_h),
            "viewBox": f"0 0 {_fmt(t.canvas_w)} {_fmt(t.canvas_h)}",
        },
    )
    defs = ET.SubElement(root, "defs")
    transparent = getattr(style, "transparent", False)
    paper = None if transparent else getattr(style, "paper", None)
    if transparent:
        bg_fill = None
    elif paper and paper[0] != paper[1]:
        grad = ET.SubElement(defs, "linearGradient",
                             {"id": "paper", "x1": "0", "y1": "0", "x2": "0", "y2": "1"})
        ET.SubElement(grad, "stop", {"offset": "0", "stop-color": paper[0]})
        ET.SubElement(grad, "stop", {"offset": "1", "stop-color": paper[1]})
        bg_fill = "url(#paper)"
    else:
        bg_fill = paper[0] if paper else style.background
    if bg_fill is not None:
        ET.SubElement(root, "rect", {"x": "0", "y": "0", "width": _fmt(t.canvas_w),
                                     "height": _fmt(t.canvas_h), "fill": bg_fill})

    for it in scene.items:
        if isinstance(it, Curve):
            cpts = t.to_canvas_arr(it.pts)
            el = ET.SubElement(root, "path", {"d": _path_d(cpts, it.closed), "fill": "none"})
            _add_stroke(el, style, it.role, it.width_scale)
            if it.color is not None:
                el.set("stroke", it.color)
            if it.dash is not None:
                d = style.dash(it.dash)
                if d is None:
                    el.attrib.pop("stroke-dasharray", None)
                else:
                    el.set("stroke-dasharray", d)
            if it.opacity < 1.0:
                el.set("stroke-opacity", _fmt(it.opacity))
            if it.arrows:
                mpts = np.vstack([cpts, cpts[:1]]) if it.closed else cpts
                color = it.color or style.ink(it.role).color
                for frac in it.arrows:
                    tip, direction = _at_fraction(mpts, frac)
                    head = _arrowhead(tip, direction, style, it.arrow_scale)
                    _emit_head(root, head, color, style,
                               hollow=(it.arrow_style == "hollow"),
                               stroke_width=style.ink(it.role).width * it.width_scale)

        elif isinstance(it, FilledCurve):
            ink = style.ink(it.role)
            attrs = {"d": _path_d(t.to_canvas_arr(it.pts), closed=True),
                     "fill": it.color or ink.color, "fill-opacity": _fmt(it.opacity)}
            el = ET.SubElement(root, "path", attrs)
            if it.edge_color is not None:
                el.set("stroke", it.edge_color)
                el.set("stroke-width", _fmt(it.edge_width or 0.4))
                el.set("stroke-linejoin", "round")
            elif it.outline:
                _add_stroke(el, style, it.role)
            else:
                el.set("stroke", "none")

        elif isinstance(it, Vector):
            tail = t.to_canvas(it.tail)
            tip = t.to_canvas(it.tip)
            d = (tip[0] - tail[0], tip[1] - tail[1])
            n = math.hypot(*d) or 1.0
            # shorten shaft so it doesn't poke through the head
            hl = style.arrowhead_len * 0.85
            shaft_end = (tip[0] - hl * d[0] / n, tip[1] - hl * d[1] / n)
            el = ET.SubElement(root, "path", {
                "d": f"M {_fmt(tail[0])} {_fmt(tail[1])} L {_fmt(shaft_end[0])} {_fmt(shaft_end[1])}",
                "fill": "none"})
            _add_stroke(el, style, it.role, it.width_scale)
            head = _arrowhead(tip, d, style)
            _emit_head(root, head, style.ink(it.role).color, style,
                       hollow=not it.filled,
                       stroke_width=style.ink(it.role).width * it.width_scale)

        elif isinstance(it, Point):
            cx, cy = t.to_canvas(it.xy)
            ink = style.ink(it.role)
            attrs = {"cx": _fmt(cx), "cy": _fmt(cy), "r": _fmt(style.point_radius * it.radius_scale)}
            if it.filled:
                attrs["fill"] = ink.color
            else:
                attrs.update({"fill": "none" if transparent else style.background,
                              "stroke": ink.color,
                              "stroke-width": _fmt(ink.width)})
            ET.SubElement(root, "circle", attrs)

        elif isinstance(it, RightAngleMark):
            c = np.array(it.corner, dtype=float)
            u1 = np.array(it.dir1, dtype=float)
            u2 = np.array(it.dir2, dtype=float)
            u1 = u1 / (np.linalg.norm(u1) or 1.0)
            u2 = u2 / (np.linalg.norm(u2) or 1.0)
            s = it.size
            pts = np.array([c + s * u1, c + s * (u1 + u2), c + s * u2])
            el = ET.SubElement(root, "path", {"d": _path_d(t.to_canvas_arr(pts), closed=False), "fill": "none"})
            _add_stroke(el, style, it.role)

        elif isinstance(it, AngleMark):
            c = np.array(it.center, dtype=float)
            a1 = math.atan2(it.dir1[1], it.dir1[0])
            a2 = math.atan2(it.dir2[1], it.dir2[0])
            if a2 < a1:
                a2 += 2 * math.pi
            arc = np.linspace(a1, a2, 32)
            pts = c + it.radius * np.column_stack([np.cos(arc), np.sin(arc)])
            el = ET.SubElement(root, "path", {"d": _path_d(t.to_canvas_arr(pts), closed=False), "fill": "none"})
            _add_stroke(el, style, it.role)

        elif isinstance(it, MathLabel):
            x, y = t.to_canvas(it.anchor)
            x += it.offset_px[0]
            y += it.offset_px[1]
            draw_math(root, it.latex, x, y,
                      size_pt=it.size_pt or style.label_size_pt,
                      color=it.color or style.ink(it.role).color,
                      halign=it.ha, valign=it.va)

    grain = 0.0 if transparent else getattr(style, "grain", 0.0)
    if grain > 0:
        from .theme import grain_tile_datauri
        pat = ET.SubElement(defs, "pattern", {
            "id": "grain", "patternUnits": "userSpaceOnUse",
            "width": "140", "height": "140"})
        ET.SubElement(pat, "image", {
            "href": grain_tile_datauri(), "width": "140", "height": "140"})
        ET.SubElement(root, "rect", {
            "x": "0", "y": "0", "width": _fmt(t.canvas_w), "height": _fmt(t.canvas_h),
            "fill": "url(#grain)", "opacity": _fmt(grain)})
    return root, t


def to_svg(scene: Scene, style: Style = DEFAULT_STYLE, width_px: float = 900) -> str:
    root, _ = to_svg_tree(scene, style, width_px)
    return ET.tostring(root, encoding="unicode")


def save(scene: Scene, path_stem: str | Path, style: Style = DEFAULT_STYLE,
         width_px: float = 900, png_scale: float = 2.0) -> tuple[Path, Path]:
    import cairosvg

    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    svg = to_svg(scene, style, width_px)
    svg_path = stem.with_suffix(".svg")
    png_path = stem.with_suffix(".png")
    svg_path.write_text(svg)
    cairosvg.svg2png(bytestring=svg.encode(), write_to=str(png_path), scale=png_scale)
    return svg_path, png_path
