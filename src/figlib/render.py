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


def _arrowhead(tip: tuple[float, float], direction: tuple[float, float], style: Style) -> list[tuple[float, float]]:
    dx, dy = direction
    n = math.hypot(dx, dy) or 1.0
    ux, uy = dx / n, dy / n
    px, py = -uy, ux
    L, W = style.arrowhead_len, style.arrowhead_halfwidth
    bx, by = tip[0] - L * ux, tip[1] - L * uy
    return [tip, (bx + W * px, by + W * py), (bx - W * px, by - W * py)]


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
    ET.SubElement(root, "rect", {"x": "0", "y": "0", "width": _fmt(t.canvas_w),
                                 "height": _fmt(t.canvas_h), "fill": style.background})

    for it in scene.items:
        if isinstance(it, Curve):
            el = ET.SubElement(root, "path", {"d": _path_d(t.to_canvas_arr(it.pts), it.closed), "fill": "none"})
            _add_stroke(el, style, it.role, it.width_scale)
            if it.color is not None:
                el.set("stroke", it.color)
            if it.opacity < 1.0:
                el.set("stroke-opacity", _fmt(it.opacity))

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
            ET.SubElement(root, "polygon", {
                "class": "arrowhead",
                "points": " ".join(f"{_fmt(x)},{_fmt(y)}" for x, y in head),
                "fill": style.ink(it.role).color})

        elif isinstance(it, Point):
            cx, cy = t.to_canvas(it.xy)
            ink = style.ink(it.role)
            attrs = {"cx": _fmt(cx), "cy": _fmt(cy), "r": _fmt(style.point_radius * it.radius_scale)}
            if it.filled:
                attrs["fill"] = ink.color
            else:
                attrs.update({"fill": style.background, "stroke": ink.color,
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
