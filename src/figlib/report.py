"""Textual scene inventory: layout proprioception without rendering a PNG.

The mechanical gate answers pass/fail; this answers "where is everything?"
— canvas bboxes for every label, grouped extents for geometry, margins,
and nearest-neighbor gaps — so layout can be debugged in text before
spending a look at the image. Coordinates are canvas px, +y down: the
same frame offset_px moves in.
"""

from __future__ import annotations

import math

from .gates import LabelBox, _box_at, _canvas_label_box, _label_boxes
from .layout import Transform
from .scene import (AngleMark, Curve, FilledCurve, MathLabel, Point,
                    RightAngleMark, Scene, Vector)
from .style import Style

BBox = tuple[float, float, float, float]


def _union(a: BBox | None, b: BBox) -> BBox:
    if a is None:
        return b
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _rect_gap(a: BBox, b: BBox) -> float:
    dx = max(b[0] - a[2], a[0] - b[2], 0.0)
    dy = max(b[1] - a[3], a[1] - b[3], 0.0)
    return math.hypot(dx, dy)


def _fmt_box(b: BBox) -> str:
    return f"({b[0]:.0f},{b[1]:.0f})..({b[2]:.0f},{b[3]:.0f})"


def _item_bbox(it, t: Transform) -> BBox | None:
    """Canvas bbox of a non-label item's geometry (marks approximated)."""
    if isinstance(it, (Curve, FilledCurve)):
        pts = t.to_canvas_arr(it.pts)
        return (float(pts[:, 0].min()), float(pts[:, 1].min()),
                float(pts[:, 0].max()), float(pts[:, 1].max()))
    if isinstance(it, Vector):
        (x0, y0), (x1, y1) = t.to_canvas(it.tail), t.to_canvas(it.tip)
        return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    if isinstance(it, Point):
        x, y = t.to_canvas(it.xy)
        return (x - 3, y - 3, x + 3, y + 3)
    if isinstance(it, (RightAngleMark, AngleMark)):
        cx, cy = it.corner if isinstance(it, RightAngleMark) else it.center
        r = (it.size if isinstance(it, RightAngleMark) else it.radius) * t.scale
        x, y = t.to_canvas((cx, cy))
        return (x - r, y - r, x + r, y + r)
    return None


def _geometry_lines(scene: Scene, t: Transform) -> tuple[list[str], BBox | None]:
    """One line per (type, role) group: count and combined extents."""
    groups: dict[tuple[str, str], tuple[int, BBox]] = {}
    ink: BBox | None = None
    for it in scene.items:
        if isinstance(it, MathLabel):
            continue
        box = _item_bbox(it, t)
        if box is None:
            continue
        key = (type(it).__name__, getattr(it.role, "name", str(it.role)))
        n, acc = groups.get(key, (0, box))
        groups[key] = (n + 1, _union(acc if n else None, box) if n else box)
        ink = _union(ink, box)
    lines = [f"  {typ} {role} x{n:<5d} {_fmt_box(box)}"
             for (typ, role), (n, box) in sorted(groups.items())]
    return lines, ink


def _label_lines(boxes: list[LabelBox]) -> list[str]:
    lines = []
    for i, (latex, pt, box) in enumerate(boxes):
        others = [(boxes[j][0], _rect_gap(box, boxes[j][2]))
                  for j in range(len(boxes)) if j != i]
        near = ""
        if others:
            name, gap = min(others, key=lambda p: p[1])
            near = (f"  nearest '{name}' {gap:.0f}px" if gap > 0
                    else f"  OVERLAPS '{name}'")
        shown = latex if len(latex) <= 28 else latex[:25] + "..."
        lines.append(f"  '{shown}' {pt:g}pt {_fmt_box(box)}{near}")
    return lines


def _scene_section(scene: Scene, style: Style, t: Transform,
                   header: str) -> list[str]:
    geo, ink = _geometry_lines(scene, t)
    boxes = _label_boxes(scene, style, t)
    out = [header, f"  canvas {t.canvas_w:.0f} x {t.canvas_h:.0f} px"]
    if ink is not None:
        out.append(
            f"  ink bbox {_fmt_box(ink)}  margins "
            f"L{ink[0]:.0f} T{ink[1]:.0f} "
            f"R{t.canvas_w - ink[2]:.0f} B{t.canvas_h - ink[3]:.0f}")
    out += ["  geometry (grouped):"] + [" " + l for l in geo]
    if boxes:
        area = sum((b[2] - b[0]) * (b[3] - b[1]) for _, _, b in boxes)
        frac = area / (t.canvas_w * t.canvas_h)
        out += [f"  labels ({len(boxes)}, {frac:.0%} of canvas; "
                f"coords are canvas px, +y down):"]
        out += [" " + l for l in _label_lines(boxes)]
    return out


def scene_report(scene: Scene, style: Style, width_px: float) -> str:
    t = Transform(scene, width_px=width_px)
    return "\n".join(_scene_section(scene, style, t, "scene"))


def figure_report(fig, style: Style, width_px: float) -> str:
    from .figure import connector_ink, layout_figure
    lay = layout_figure(fig, width_px)
    out = [f"figure  canvas {lay.canvas_w:.0f} x {lay.canvas_h:.0f} px, "
           f"{len(fig.panels)} panels"]
    for i, (panel, slot) in enumerate(zip(fig.panels, lay.slots)):
        tag = f" tag {panel.tag}" if panel.tag else ""
        out += _scene_section(
            panel.scene, style, slot.transform,
            f"panel[{i}]{tag}  slot x{slot.x:.0f} y{slot.y:.0f} "
            f"w{slot.w:.0f} h{slot.h:.0f} (label coords panel-local)")
    page: list[LabelBox] = [_canvas_label_box(lab, style) for lab in fig.page_items]
    for conn in fig.connectors:
        page += [_canvas_label_box(lab, style)
                 for lab in connector_ink(conn, lay, style).labels]
    if page:
        out += ["page labels (figure canvas px):"] + _label_lines(page)
    return "\n".join(out)


def report(built, style: Style, width_px: float) -> str:
    """Dispatch on what build() returned."""
    if isinstance(built, Scene):
        return scene_report(built, style, width_px)
    return figure_report(built, style, width_px)
