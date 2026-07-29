"""SVG emission and PNG rasterization.

Arrowheads are explicit filled polygons (not markers) so every piece of
ink has a bbox we own.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from .layout import Transform, geometry_extents
from .scene import (AngleMark, Brace, Callout, Curve, FilledCurve, Gradient,
                    MathLabel, Point, RasterField, RightAngleMark, Scene,
                    Vector)
from .style import DEFAULT_STYLE, Role, Style
from .typeset import draw_math, render_math

SVG_NS = "http://www.w3.org/2000/svg"

# PNG raster scale: png px = canvas px * PNG_SCALE. Anything converting
# between the two frames (figcheck --zoom) must use this, not a literal.
PNG_SCALE = 2.0


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


def _vertex_normals(pts: np.ndarray) -> np.ndarray:
    """Unit normals per vertex: average of adjacent segment normals."""
    seg = np.diff(pts, axis=0)
    lens = np.hypot(seg[:, 0], seg[:, 1])
    lens[lens == 0.0] = 1e-12
    n_seg = np.column_stack([-seg[:, 1], seg[:, 0]]) / lens[:, None]
    n_v = np.vstack([n_seg[:1], n_seg[:-1] + n_seg[1:], n_seg[-1:]])
    mag = np.hypot(n_v[:, 0], n_v[:, 1])
    mag[mag == 0.0] = 1e-12
    return n_v / mag[:, None]


def _var_width_polygon(pts: np.ndarray, widths: np.ndarray) -> np.ndarray:
    """Offset polygon for a stroke whose width varies per vertex (px)."""
    normals = _vertex_normals(pts)
    half = (widths / 2.0)[:, None] * normals
    return np.vstack([pts + half, (pts - half)[::-1]])


def _profile_widths(profile: tuple[float, ...], pts: np.ndarray, base: float) -> np.ndarray:
    """Per-vertex widths: profile at equal arc-length knots, interpolated."""
    seg = np.diff(pts, axis=0)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))])
    t = s / (s[-1] or 1.0)
    knots = np.linspace(0.0, 1.0, max(len(profile), 2))
    vals = np.array(profile if len(profile) > 1 else (profile[0], profile[0]))
    return base * np.interp(t, knots, vals)


def _svg_root(w: float, h: float) -> ET.Element:
    return ET.Element("svg", {
        "xmlns": SVG_NS, "width": _fmt(w), "height": _fmt(h),
        "viewBox": f"0 0 {_fmt(w)} {_fmt(h)}"})


def _emit_paper(root: ET.Element, defs: ET.Element, style: Style, w: float, h: float) -> None:
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
        ET.SubElement(root, "rect", {"x": "0", "y": "0", "width": _fmt(w),
                                     "height": _fmt(h), "fill": bg_fill})


def _ensure_grain_pattern(defs: ET.Element) -> str:
    """The shared grain tile pattern def (page overlay + fill overlays)."""
    if defs.find("./pattern[@id='grain']") is None:
        from .theme import grain_tile_datauri
        pat = ET.SubElement(defs, "pattern", {
            "id": "grain", "patternUnits": "userSpaceOnUse",
            "width": "140", "height": "140"})
        ET.SubElement(pat, "image", {
            "href": grain_tile_datauri(), "width": "140", "height": "140"})
    return "grain"


def _emit_grain(root: ET.Element, defs: ET.Element, style: Style, w: float, h: float) -> None:
    grain = 0.0 if getattr(style, "transparent", False) else getattr(style, "grain", 0.0)
    if grain > 0:
        ET.SubElement(root, "rect", {
            "x": "0", "y": "0", "width": _fmt(w), "height": _fmt(h),
            "fill": f"url(#{_ensure_grain_pattern(defs)})", "opacity": _fmt(grain)})


# --- annotation glyphs (geometry shared with the mechanical gate) -----------

# Label clearance beyond the brace cusp / callout box padding (canvas px).
BRACE_LABEL_GAP = 6.0
CALLOUT_PAD = 5.0
CALLOUT_RADIUS = 6.0
CALLOUT_HEAD_SCALE = 0.55


def _outward_alignment(n: np.ndarray) -> tuple[str, str]:
    """ha/va that place a label on the far side of its anchor along the
    outward canvas direction n (+y down)."""
    if abs(n[0]) > abs(n[1]):
        return ("left", "center") if n[0] > 0 else ("right", "center")
    return ("center", "top") if n[1] > 0 else ("center", "bottom")


# Path skeleton of the brace's 15 control points (index -> segment role):
#   M 0 | C 1 2 3 | L 4 | C 5 6 7 | C 8 9 10 | L 11 | C 12 13 14
# with the center cusp at index 7.
BRACE_CUSP = 7


def brace_ink(it: Brace, t: Transform, style: Style) -> tuple[np.ndarray, MathLabel | None]:
    """Resolve a Brace to canvas px: the TeX brace — two mirrored halves
    (end hook, straight shoulder at half depth, turn up to the cusp) as
    four cubics + two line segments, 15 control points with the cusp at
    index BRACE_CUSP. Control points are computed in math coords and
    mapped pointwise; cubics are affine-invariant, so the drawn curve is
    exact. Returns (control points, label past the cusp or None)."""
    mid, u, n, d = it.frame()
    p1 = np.asarray(it.p1, dtype=float)
    span = float(np.hypot(*(np.asarray(it.p2, dtype=float) - p1)))
    L, m = span, d / 2.0
    r = min(m, L / 4.0)                       # hook radius along the span

    def P(x: float, y: float) -> np.ndarray:
        return p1 + x * u + y * n

    # quadratic corners converted to cubics (c1 = p0 + 2/3(q-p0), ...)
    ctrl = np.array([
        P(0, 0),                                            # 0  M end 1
        P(0, 2 * m / 3), P(r / 3, m), P(r, m),              # 1-3  C hook up
        P(L / 2 - r, m),                                    # 4  L shoulder
        P(L / 2 - r / 3, m), P(L / 2, d - 2 * m / 3),       # 5-6
        P(L / 2, d),                                        # 7  cusp
        P(L / 2, d - 2 * m / 3), P(L / 2 + r / 3, m),       # 8-9
        P(L / 2 + r, m),                                    # 10 C down from cusp
        P(L - r, m),                                        # 11 L shoulder
        P(L - r / 3, m), P(L, 2 * m / 3), P(L, 0),          # 12-14 C hook down
    ])
    cpts = t.to_canvas_arr(ctrl)
    label = None
    if it.label is not None:
        out = cpts[BRACE_CUSP] - t.to_canvas_arr(mid[None, :])[0]  # canvas +y down
        out = out / (float(np.hypot(*out)) or 1.0)
        ax, ay = cpts[BRACE_CUSP] + BRACE_LABEL_GAP * out
        ha, va = _outward_alignment(out)
        label = MathLabel(it.label, (float(ax), float(ay)), role=it.role,
                          ha=ha, va=va)
    return cpts, label


@dataclass
class CalloutInk:
    """Callout geometry in canvas px — render draws it, the gate boxes it."""
    box: tuple[float, float, float, float]   # x0, y0, x1, y1 (label + padding)
    leader: np.ndarray                       # (2, 2): box edge -> target
    head: list[tuple[float, float]]          # filled arrowhead at the target
    label: MathLabel                         # anchor in canvas px


def callout_ink(it: Callout, t: Transform, style: Style) -> CalloutInk:
    """Box centered on the anchor; leader leaves from the boundary point
    nearest the target (clamp of the target to the box)."""
    pt = style.label_pt(it.size_pt)
    m = render_math(it.latex, pt)
    cx, cy = t.to_canvas(it.anchor)
    hw = m.width_px / 2 + CALLOUT_PAD
    hh = m.height_px / 2 + CALLOUT_PAD
    box = (cx - hw, cy - hh, cx + hw, cy + hh)
    tx, ty = t.to_canvas(it.target)
    ex = min(max(tx, box[0]), box[2])
    ey = min(max(ty, box[1]), box[3])
    leader = np.array([[ex, ey], [tx, ty]])
    head = _arrowhead((tx, ty), (tx - ex, ty - ey), style, scale=CALLOUT_HEAD_SCALE)
    label = MathLabel(it.latex, (cx, cy), role=it.role, size_pt=it.size_pt,
                      ha="center", va="center")
    return CalloutInk(box, leader, head, label)


# --- pattern fills and raster fields ----------------------------------------

_PATTERN_TILE = {"stipple": 5.0, "hatch": 6.0, "crosshatch": 6.0}


def _ensure_pattern(defs: ET.Element, pattern: str, color: str) -> str:
    """One <pattern> def per (pattern, color) pair, reused by every fill
    that asks for it. Tile ground is transparent — paper shows through."""
    pid = f"pat-{pattern}-{str(color).lstrip('#')}"
    if defs.find(f"./pattern[@id='{pid}']") is not None:
        return pid
    s = _PATTERN_TILE[pattern]
    attrs = {"id": pid, "patternUnits": "userSpaceOnUse",
             "width": _fmt(s), "height": _fmt(s)}
    if pattern in ("hatch", "crosshatch"):
        attrs["patternTransform"] = "rotate(45)"
    pat = ET.SubElement(defs, "pattern", attrs)
    if pattern == "stipple":
        ET.SubElement(pat, "circle", {"cx": _fmt(s / 2), "cy": _fmt(s / 2),
                                      "r": "0.80", "fill": color})
    else:
        line = {"stroke": color, "stroke-width": "0.80"}
        # lines at the tile center, fully interior, so tiling has no seams
        ET.SubElement(pat, "line", {"x1": _fmt(s / 2), "y1": "0",
                                    "x2": _fmt(s / 2), "y2": _fmt(s), **line})
        if pattern == "crosshatch":
            ET.SubElement(pat, "line", {"x1": "0", "y1": _fmt(s / 2),
                                        "x2": _fmt(s), "y2": _fmt(s / 2), **line})
    return pid


def _ensure_gradient(defs: ET.Element, grad: Gradient, t: Transform) -> str:
    """One gradient def per (kind, stops, canvas axis), content-addressed."""
    x0, y0 = t.to_canvas(grad.p0)
    x1, y1 = t.to_canvas(grad.p1)
    key = (grad.kind, grad.stops, round(x0, 2), round(y0, 2),
           round(x1, 2), round(y1, 2))
    gid = "grad-" + hashlib.md5(repr(key).encode()).hexdigest()[:10]
    tag = "linearGradient" if grad.kind == "linear" else "radialGradient"
    if defs.find(f"./{tag}[@id='{gid}']") is not None:
        return gid
    if grad.kind == "linear":
        el = ET.SubElement(defs, "linearGradient", {
            "id": gid, "gradientUnits": "userSpaceOnUse",
            "x1": _fmt(x0), "y1": _fmt(y0), "x2": _fmt(x1), "y2": _fmt(y1)})
    elif grad.kind == "radial":
        el = ET.SubElement(defs, "radialGradient", {
            "id": gid, "gradientUnits": "userSpaceOnUse",
            "cx": _fmt(x0), "cy": _fmt(y0),
            "r": _fmt(math.hypot(x1 - x0, y1 - y0))})
    else:
        raise ValueError(f"unknown gradient kind {grad.kind!r}")
    for off, color in grad.stops:
        ET.SubElement(el, "stop", {"offset": _fmt(off), "stop-color": color})
    return gid


def _hex_rgb(color: str) -> tuple[int, int, int]:
    c = str(color).lstrip("#")
    return tuple(int(c[k:k + 2], 16) for k in (0, 2, 4))


def _gray_ramp(t: float) -> str:
    """Fallback ramp: light (t=0) to near-black ink (t=1)."""
    g = round(240 + t * (26 - 240))
    return f"#{g:02x}{g:02x}{g:02x}"


def _rgb_datauri(rgb: "np.ndarray", interp: bool) -> str:
    """(H, W, 3) uint8 -> opaque RGBA PNG data URI, with the same
    nearest-neighbor upsample the scalar path uses when interp is off."""
    import base64
    import io

    from PIL import Image

    if not interp:
        # rasterizers that ignore image-rendering:pixelated (cairosvg) still
        # show the cell grid
        k = max(1, min(8, 2048 // max(rgb.shape[:2])))
        rgb = np.repeat(np.repeat(rgb, k, axis=0), k, axis=1)
    rgba = np.dstack([rgb, np.full(rgb.shape[:2], 255, dtype=np.uint8)])
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _raster_datauri(it: RasterField, style: Style) -> str:
    """values -> normalized [0,1] -> 256-level ramp LUT -> RGBA PNG.

    An (H, W, 3) `values` array skips all of that: those pixels are already
    colors, because the field's value was never a scalar. That is the case
    for a cyclic channel — phase is periodic, so no monotone ramp can carry
    it and `theme.phase` resolves the color itself, per pixel.
    """
    v = np.asarray(it.values, dtype=float)
    if v.ndim == 3:
        if v.shape[2] != 3:
            raise ValueError(
                f"RasterField values with ndim 3 must be (H, W, 3) sRGB in "
                f"[0, 1]; got shape {v.shape}")
        return _rgb_datauri(np.round(np.clip(v, 0.0, 1.0) * 255).astype(np.uint8),
                            it.interp)
    vmin = it.vmin if it.vmin is not None else float(np.nanmin(v))
    vmax = it.vmax if it.vmax is not None else float(np.nanmax(v))
    span = (vmax - vmin) or 1.0
    t01 = np.clip((v - vmin) / span, 0.0, 1.0)
    ramp = it.ramp or getattr(style, "ramp", None) or _gray_ramp
    lut = np.array([_hex_rgb(ramp(k / 255.0)) for k in range(256)], dtype=np.uint8)
    idx = np.round(t01 * 255.0).astype(np.uint8)
    return _rgb_datauri(lut[idx], it.interp)


def _emit_clip(parent: ET.Element, scene: Scene, defs: ET.Element,
               t: Transform) -> ET.Element:
    """One <clipPath> for the scene: the xlim/ylim rect ("frame") or a
    math-coords polygon (clip_pts). Returns the group items draw into."""
    cid = f"clip-{len(defs.findall('clipPath'))}"
    cp = ET.SubElement(defs, "clipPath", {"id": cid})
    if scene.clip_pts is not None:
        pts = t.to_canvas_arr(np.asarray(scene.clip_pts, dtype=float))
        ET.SubElement(cp, "path", {"d": _path_d(pts, closed=True)})
    else:  # "frame"
        (x0, x1), (y0, y1) = geometry_extents(scene)
        cx0, cy0 = t.to_canvas((x0, y1))
        cx1, cy1 = t.to_canvas((x1, y0))
        ET.SubElement(cp, "rect", {"x": _fmt(cx0), "y": _fmt(cy0),
                                   "width": _fmt(cx1 - cx0),
                                   "height": _fmt(cy1 - cy0)})
    return ET.SubElement(parent, "g", {"clip-path": f"url(#{cid})"})


def _emit_items(parent: ET.Element, scene: Scene, style: Style, t: Transform,
                defs: ET.Element | None = None) -> None:
    """Emit every scene item into parent — the one per-item loop, shared by
    single-scene and multi-panel emission. defs receives shared resources
    (pattern tiles, clip paths)."""
    transparent = getattr(style, "transparent", False)
    root = parent
    if defs is not None and (scene.clip is not None or scene.clip_pts is not None):
        root = _emit_clip(parent, scene, defs, t)
    for it in scene.items:
        if isinstance(it, Curve):
            cpts = t.to_canvas_arr(it.pts)
            if it.width_profile is not None:
                spts = np.vstack([cpts, cpts[:1]]) if it.closed else cpts
                ink = style.ink(it.role)
                widths = _profile_widths(it.width_profile, spts,
                                         ink.width * it.width_scale)
                poly = _var_width_polygon(spts, widths)
                el = ET.SubElement(root, "path", {
                    "class": "varstroke", "d": _path_d(poly, closed=True),
                    "fill": it.color or ink.color})
                if it.opacity < 1.0:
                    el.set("fill-opacity", _fmt(it.opacity))
            else:
                if it.casing and not transparent:
                    ink = style.ink(it.role)
                    ET.SubElement(root, "path", {
                        "d": _path_d(cpts, it.closed), "fill": "none",
                        "stroke": style.background,
                        "stroke-width": _fmt(ink.width * it.width_scale
                                             + 2.0 * style.halo_width),
                        "stroke-linecap": "round", "stroke-linejoin": "round"})
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
                if it.cap is not None:
                    el.set("stroke-linecap", it.cap)
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
            fill_color = it.color or ink.color
            d = _path_d(t.to_canvas_arr(it.pts), closed=True)
            for hole in it.holes:
                d += " " + _path_d(t.to_canvas_arr(hole), closed=True)
            attrs = {"d": d}
            if it.holes:
                attrs["fill-rule"] = "evenodd"
            if it.gradient is not None and defs is not None:
                attrs["fill"] = f"url(#{_ensure_gradient(defs, it.gradient, t)})"
                if it.opacity < 1.0:
                    attrs["fill-opacity"] = _fmt(it.opacity)
            elif it.pattern is not None and defs is not None:
                # texture is ink, not a wash — item opacity does not apply
                attrs["fill"] = f"url(#{_ensure_pattern(defs, it.pattern, fill_color)})"
            else:
                attrs["fill"] = fill_color
                attrs["fill-opacity"] = _fmt(it.opacity)
            el = ET.SubElement(root, "path", attrs)
            if it.edge_color is not None:
                el.set("stroke", it.edge_color)
                el.set("stroke-width", _fmt(it.edge_width or 0.4))
                el.set("stroke-linejoin", "round")
            elif it.outline:
                _add_stroke(el, style, it.role)
            else:
                el.set("stroke", "none")
            if it.grain > 0 and not transparent and defs is not None:
                gattrs = {"d": d,
                          "fill": f"url(#{_ensure_grain_pattern(defs)})",
                          "fill-opacity": _fmt(it.grain), "stroke": "none"}
                if it.holes:
                    gattrs["fill-rule"] = "evenodd"
                ET.SubElement(root, "path", gattrs)

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
            color = it.color or ink.color
            attrs = {"cx": _fmt(cx), "cy": _fmt(cy), "r": _fmt(style.point_radius * it.radius_scale)}
            if it.filled:
                attrs["fill"] = color
            else:
                attrs.update({"fill": "none" if transparent else style.background,
                              "stroke": color,
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

        elif isinstance(it, Brace):
            cpts, label = brace_ink(it, t, style)
            p = [f"{_fmt(x)} {_fmt(y)}" for x, y in cpts]
            d = (f"M {p[0]} C {p[1]} {p[2]} {p[3]} L {p[4]} "
                 f"C {p[5]} {p[6]} {p[7]} C {p[8]} {p[9]} {p[10]} "
                 f"L {p[11]} C {p[12]} {p[13]} {p[14]}")
            el = ET.SubElement(root, "path", {"class": "brace", "d": d,
                                              "fill": "none"})
            _add_stroke(el, style, it.role)
            if label is not None:
                _emit_canvas_label(root, label, style)

        elif isinstance(it, Callout):
            cink = callout_ink(it, t, style)
            color = style.ink(it.role).color
            el = ET.SubElement(root, "path", {
                "d": _path_d(cink.leader, closed=False), "fill": "none"})
            _add_stroke(el, style, it.role)
            _emit_head(root, cink.head, color, style, hollow=False,
                       stroke_width=style.ink(it.role).width)
            if it.boxed:
                x0, y0, x1, y1 = cink.box
                paper = getattr(style, "paper", None)
                frame_w = style.ink(Role.FRAME).width
                ET.SubElement(root, "rect", {
                    "class": "callout-box",
                    "x": _fmt(x0), "y": _fmt(y0),
                    "width": _fmt(x1 - x0), "height": _fmt(y1 - y0),
                    "rx": _fmt(CALLOUT_RADIUS),
                    "fill": "none" if transparent
                            else (paper[0] if paper else style.background),
                    "stroke": color, "stroke-width": _fmt(frame_w)})
            _emit_canvas_label(root, cink.label, style)

        elif isinstance(it, RasterField):
            x0, x1, y0, y1 = it.extent
            px, py = t.to_canvas((x0, y1))          # row 0 = top = y1
            attrs = {"href": _raster_datauri(it, style),
                     "x": _fmt(px), "y": _fmt(py),
                     "width": _fmt((x1 - x0) * t.scale_x),
                     "height": _fmt((y1 - y0) * t.scale_y),
                     "preserveAspectRatio": "none"}
            if not it.interp:
                attrs["image-rendering"] = "pixelated"
            if it.opacity < 1.0:
                attrs["opacity"] = _fmt(it.opacity)
            ET.SubElement(root, "image", attrs)

        elif isinstance(it, MathLabel):
            x, y = t.to_canvas(it.anchor)
            x += it.offset_px[0]
            y += it.offset_px[1]
            tgt = root
            if it.angle_deg:
                # angle_deg is CCW on the page; SVG's rotate() is CW (+y down)
                tgt = ET.SubElement(root, "g", {
                    "transform": f"rotate({_fmt(-it.angle_deg)} {_fmt(x)} {_fmt(y)})"})
            halo_on = it.halo and not transparent
            draw_math(tgt, it.latex, x, y,
                      size_pt=style.label_pt(it.size_pt),
                      color=it.color or style.ink(it.role).color,
                      halign=it.ha, valign=it.va,
                      halo_color=style.background if halo_on else None,
                      halo_width=style.halo_width if halo_on else 0.0)


def to_svg_tree(scene: Scene, style: Style = DEFAULT_STYLE, width_px: float = 900) -> tuple[ET.Element, Transform]:
    t = Transform(scene, width_px=width_px)
    root = _svg_root(t.canvas_w, t.canvas_h)
    defs = ET.SubElement(root, "defs")
    _emit_paper(root, defs, style, t.canvas_w, t.canvas_h)
    _emit_items(root, scene, style, t, defs)
    _emit_grain(root, defs, style, t.canvas_w, t.canvas_h)
    return root, t


def to_svg(scene: Scene, style: Style = DEFAULT_STYLE, width_px: float = 900) -> str:
    root, _ = to_svg_tree(scene, style, width_px)
    return ET.tostring(root, encoding="unicode")


def _emit_canvas_label(root: ET.Element, lab: MathLabel, style: Style) -> None:
    """A MathLabel whose anchor is already in figure canvas px (page layer,
    connector labels, glyph labels)."""
    x = lab.anchor[0] + lab.offset_px[0]
    y = lab.anchor[1] + lab.offset_px[1]
    if lab.angle_deg:
        root = ET.SubElement(root, "g", {
            "transform": f"rotate({_fmt(-lab.angle_deg)} {_fmt(x)} {_fmt(y)})"})
    draw_math(root, lab.latex, x, y,
              size_pt=style.label_pt(lab.size_pt),
              color=lab.color or style.ink(lab.role).color,
              halign=lab.ha, valign=lab.va)


def figure_svg_tree(fig: "Figure", style: Style = DEFAULT_STYLE,
                    width_px: float = 900) -> tuple[ET.Element, list[Transform]]:
    """Multi-panel emission: one paper + grain for the whole canvas, each
    panel in its own translated group, frames/tags/connectors/page items
    at figure level."""
    from .figure import (FRAME_INSET, FRAME_RADIUS, TAG_PAD, connector_ink,
                         layout_figure)

    lay = layout_figure(fig, width_px)
    root = _svg_root(lay.canvas_w, lay.canvas_h)
    defs = ET.SubElement(root, "defs")
    _emit_paper(root, defs, style, lay.canvas_w, lay.canvas_h)

    for panel, slot in zip(fig.panels, lay.slots):
        g = ET.SubElement(root, "g", {
            "transform": f"translate({_fmt(slot.x)} {_fmt(slot.content_y)})"})
        # clip coords are panel-local: the clip group sits inside the translate
        _emit_items(g, panel.scene, style, slot.transform, defs)

    frame_ink = style.ink(Role.FRAME)
    ann_ink = style.ink(Role.ANNOTATION)
    for panel, slot in zip(fig.panels, lay.slots):
        if panel.frame:
            attrs = {"x": _fmt(slot.x + FRAME_INSET), "y": _fmt(slot.y + FRAME_INSET),
                     "width": _fmt(slot.w - 2 * FRAME_INSET),
                     "height": _fmt(slot.h - 2 * FRAME_INSET),
                     "rx": _fmt(FRAME_RADIUS), "fill": "none",
                     "stroke": frame_ink.color, "stroke-width": _fmt(frame_ink.width),
                     "stroke-linecap": "round"}
            if frame_ink.dash:
                attrs["stroke-dasharray"] = frame_ink.dash
            ET.SubElement(root, "rect", attrs)
        if panel.tag:
            draw_math(root, panel.tag,
                      slot.x + FRAME_INSET + TAG_PAD, slot.y + FRAME_INSET + TAG_PAD,
                      size_pt=style.label_size_pt, color=ann_ink.color,
                      halign="left", valign="top")

    transparent = getattr(style, "transparent", False)
    for conn in fig.connectors:
        ink = connector_ink(conn, lay, style)
        for pts in ink.strokes:
            el = ET.SubElement(root, "path", {"d": _path_d(pts, False), "fill": "none"})
            _add_stroke(el, style, Role.ANNOTATION)
        for head in ink.heads:
            ET.SubElement(root, "polygon", {
                "class": "arrowhead", "fill": ann_ink.color,
                "points": " ".join(f"{_fmt(x)},{_fmt(y)}" for x, y in head)})
        for poly in ink.hollow:
            el = ET.SubElement(root, "polygon", {
                "points": " ".join(f"{_fmt(x)},{_fmt(y)}" for x, y in poly),
                "fill": "none" if transparent else style.background})
            _add_stroke(el, style, Role.ANNOTATION)
        for lab in ink.labels:
            _emit_canvas_label(root, lab, style)

    for lab in fig.page_items:
        _emit_canvas_label(root, lab, style)

    _emit_grain(root, defs, style, lay.canvas_w, lay.canvas_h)
    return root, [s.transform for s in lay.slots]


def figure_to_svg(fig: "Figure", style: Style = DEFAULT_STYLE, width_px: float = 900) -> str:
    root, _ = figure_svg_tree(fig, style, width_px)
    return ET.tostring(root, encoding="unicode")


def save(scene: Scene, path_stem: str | Path, style: Style = DEFAULT_STYLE,
         width_px: float = 900, png_scale: float = PNG_SCALE) -> tuple[Path, Path]:
    import cairosvg

    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    svg = to_svg(scene, style, width_px)
    svg_path = stem.with_suffix(".svg")
    png_path = stem.with_suffix(".png")
    svg_path.write_text(svg)
    cairosvg.svg2png(bytestring=svg.encode(), write_to=str(png_path), scale=png_scale)
    return svg_path, png_path


def save_figure(fig: "Figure", path_stem: str | Path, style: Style = DEFAULT_STYLE,
                width_px: float = 900, png_scale: float = PNG_SCALE) -> tuple[Path, Path]:
    import cairosvg

    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    svg = figure_to_svg(fig, style, width_px)
    svg_path = stem.with_suffix(".svg")
    png_path = stem.with_suffix(".png")
    svg_path.write_text(svg)
    cairosvg.svg2png(bytestring=svg.encode(), write_to=str(png_path), scale=png_scale)
    return svg_path, png_path
