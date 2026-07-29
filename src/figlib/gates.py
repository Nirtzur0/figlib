"""Deterministic gates: numerical assertions and mechanical render checks.

The readback gate (cold-reader test) lives in readback.py — it needs a
model; these two do not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .layout import Transform
from .render import brace_ink, callout_ink
from .scene import Brace, Callout, Curve, MathLabel, Point, Scene, Vector
from .style import Role, Style
from .typeset import apply_register, render_math

# Fraction of a label's height assumed to hang below the baseline.
_DESCENT_FRAC = 0.2


@dataclass(frozen=True)
class Diagnostic:
    kind: str    # 'label-collision' | 'clipped' | 'tiny-label' | 'label-scale'
                 # | 'annotation-load' | 'numerical' | 'faint-ink' | 'hue-collapse'
                 # | 'label-on-ink' | 'arrow-on-mark'
    detail: str


# Legibility floor: below this the label is unreadable at display size.
MIN_LABEL_PT = 8.5
# A single label taller than this fraction of the canvas dominates it.
MAX_LABEL_HEIGHT_FRAC = 0.18
# Total label area beyond this fraction means the slot is too small for
# the annotation load — use a larger Format or trim, never shrink type.
MAX_ANNOTATION_AREA_FRAC = 0.22


# A label resolved to canvas space: (latex, size_pt, (x0, y0, x1, y1)).
LabelBox = tuple[str, float, tuple[float, float, float, float]]


def _box_at(latex: str, size_pt: float, x: float, y: float,
            ha: str, va: str,
            register: str | None = None) -> tuple[float, float, float, float]:
    """Exact typeset bbox anchored at canvas (x, y).

    THE measurement choke point: every label box in the library comes
    through here, so applying the register once here is what keeps the
    gate's boxes on the ink the renderer draws."""
    m = render_math(apply_register(latex, register), size_pt)
    w, h = m.width_px, m.height_px
    if ha == "center":
        x -= w / 2
    elif ha == "right":
        x -= w
    if va == "base":
        y0 = y - h * (1 - _DESCENT_FRAC)
    elif va == "top":
        y0 = y
    elif va == "center":
        y0 = y - h / 2
    else:  # bottom
        y0 = y - h
    return (x, y0, x + w, y0 + h)


def _rotate_box(box: tuple[float, float, float, float], cx: float, cy: float,
                angle_deg: float) -> tuple[float, float, float, float]:
    """AABB of a box rotated about (cx, cy) — the same rotation the render
    emits (angle_deg CCW on the page, canvas +y down)."""
    th = math.radians(-angle_deg)
    c, s = math.cos(th), math.sin(th)
    xs, ys = [], []
    for x, y in ((box[0], box[1]), (box[2], box[1]),
                 (box[2], box[3]), (box[0], box[3])):
        dx, dy = x - cx, y - cy
        xs.append(cx + c * dx - s * dy)
        ys.append(cy + s * dx + c * dy)
    return (min(xs), min(ys), max(xs), max(ys))


def _canvas_label_box(it: MathLabel, style: Style) -> LabelBox:
    """A MathLabel whose anchor is already canvas px (page layer, glyph
    labels)."""
    pt = style.label_pt(it.size_pt)
    x = it.anchor[0] + it.offset_px[0]
    y = it.anchor[1] + it.offset_px[1]
    box = _box_at(it.latex, pt, x, y, it.ha, it.va, it.register)
    if it.angle_deg:
        box = _rotate_box(box, x, y, it.angle_deg)
    return (it.latex, pt, box)


# A resolved box plus the MathLabel that owns it — None when the box is
# derived geometry (brace labels, callout boxes) that placement cannot move.
LabelEntry = tuple[LabelBox, MathLabel | None]


def _label_entries(scene: Scene, style: Style, t: Transform) -> list[LabelEntry]:
    """Every box the mechanical gate owns: MathLabels (rotation-aware),
    Brace labels, Callout boxes — resolved through the same geometry the
    render emits. Each box carries its owning MathLabel when one exists,
    so the auto-place pass knows what it may move."""
    entries: list[LabelEntry] = []
    for it in scene.items:
        if isinstance(it, MathLabel):
            pt = style.label_pt(it.size_pt)
            x, y = t.to_canvas(it.anchor)
            x += it.offset_px[0]
            y += it.offset_px[1]
            box = _box_at(it.latex, pt, x, y, it.ha, it.va, it.register)
            if it.angle_deg:
                box = _rotate_box(box, x, y, it.angle_deg)
            entries.append(((it.latex, pt, box), it))
        elif isinstance(it, Brace) and it.label is not None:
            _, lab = brace_ink(it, t, style)
            entries.append((_canvas_label_box(lab, style), None))
        elif isinstance(it, Callout):
            pt = style.label_pt(it.size_pt)
            # owner = the Callout itself: consumers must treat it as
            # pinned (anchors aren't offset-nudgeable); its paper box is
            # declared ink-cover when boxed, like halo on a MathLabel
            entries.append(((it.latex, pt, callout_ink(it, t, style).box), it))
    return entries


def _label_boxes(scene: Scene, style: Style, t: Transform) -> list[LabelBox]:
    return [box for box, _ in _label_entries(scene, style, t)]


def _overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


# Clearance added to a suggested nudge so the fix doesn't land labels flush.
_NUDGE_PAD = 3.0


def _free_nudges(k: int, boxes: list[LabelBox], canvas_w: float,
                 canvas_h: float) -> list[tuple[float, float]]:
    """Single-axis offset_px deltas (canvas px, +y down) that would move
    box k clear of every box it currently overlaps, verified against ALL
    other boxes and the canvas bounds. Sorted smallest first."""
    box = boxes[k][2]
    partners = [boxes[m][2] for m in range(len(boxes))
                if m != k and _overlap(box, boxes[m][2])]
    if not partners:
        return []
    moves = [
        (max(p[2] - box[0] for p in partners) + _NUDGE_PAD, 0.0),    # right
        (min(p[0] - box[2] for p in partners) - _NUDGE_PAD, 0.0),    # left
        (0.0, max(p[3] - box[1] for p in partners) + _NUDGE_PAD),    # down
        (0.0, min(p[1] - box[3] for p in partners) - _NUDGE_PAD),    # up
    ]
    free = []
    for dx, dy in moves:
        cand = (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)
        if cand[0] < 0 or cand[1] < 0 or cand[2] > canvas_w or cand[3] > canvas_h:
            continue
        if all(not _overlap(cand, boxes[m][2]) for m in range(len(boxes)) if m != k):
            free.append((dx, dy))
    return sorted(free, key=lambda d: abs(d[0]) + abs(d[1]))


def _fmt_nudge(d: tuple[float, float]) -> str:
    return f"({d[0]:+.0f}, {d[1]:+.0f})"


def _check_boxes(boxes: list[LabelBox], canvas_w: float, canvas_h: float,
                 type_scale: float = 1.0) -> list[Diagnostic]:
    """The label checks: collision, clipping, legibility, annotation load.

    Diagnostics carry the fix, not just the failure: collisions report the
    overlap size and verified free offset_px nudges (+y down); clipping
    reports the overrun per edge.
    """
    diags: list[Diagnostic] = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][2], boxes[j][2]
            if not _overlap(a, b):
                continue
            ox = min(a[2], b[2]) - max(a[0], b[0])
            oy = min(a[3], b[3]) - max(a[1], b[1])
            fix = ""
            for k, latex in ((j, boxes[j][0]), (i, boxes[i][0])):
                nudges = _free_nudges(k, boxes, canvas_w, canvas_h)
                if nudges:
                    opts = " or ".join(_fmt_nudge(d) for d in nudges[:2])
                    fix = f" — free: offset_px of '{latex}' += {opts}"
                    break
            if not fix:
                fix = " — no free single-axis nudge; move an anchor or enlarge the Format"
            diags.append(Diagnostic(
                "label-collision",
                f"'{boxes[i][0]}' overlaps '{boxes[j][0]}' by {ox:.0f}x{oy:.0f} px{fix}"))
    for latex, pt, (x0, y0, x1, y1) in boxes:
        if x0 < 0 or y0 < 0 or x1 > canvas_w or y1 > canvas_h:
            over = []
            if x0 < 0:
                over.append(f"{-x0:.0f}px past the left edge")
            if x1 > canvas_w:
                over.append(f"{x1 - canvas_w:.0f}px past the right edge")
            if y0 < 0:
                over.append(f"{-y0:.0f}px past the top edge")
            if y1 > canvas_h:
                over.append(f"{y1 - canvas_h:.0f}px past the bottom edge")
            diags.append(Diagnostic(
                "clipped", f"'{latex}' extends {', '.join(over)} — "
                f"nudge offset_px back in or pull the anchor inward"))
        # Legibility at READING size: the floor scales with the format's
        # ink scale so it always means the same physical type size.
        if pt < MIN_LABEL_PT * type_scale:
            diags.append(Diagnostic(
                "tiny-label", f"'{latex}' at {pt / type_scale:.1f}pt reading size "
                f"< {MIN_LABEL_PT}pt floor"))
    canvas_area = canvas_w * canvas_h
    label_area = 0.0
    for latex, pt, (x0, y0, x1, y1) in boxes:
        label_area += (x1 - x0) * (y1 - y0)
        if (y1 - y0) > MAX_LABEL_HEIGHT_FRAC * canvas_h:
            diags.append(Diagnostic(
                "label-scale",
                f"'{latex}' is {(y1 - y0) / canvas_h:.0%} of canvas height — "
                f"use a larger Format, not smaller type"))
    if canvas_area and label_area > MAX_ANNOTATION_AREA_FRAC * canvas_area:
        diags.append(Diagnostic(
            "annotation-load",
            f"labels cover {label_area / canvas_area:.0%} of the canvas "
            f"(> {MAX_ANNOTATION_AREA_FRAC:.0%}) — larger Format or less annotation"))
    return diags


# --- ink corridors: content strokes as keep-out regions ---------------------
#
# A label sitting on content ink is the #1 post-PASS defect: the gate
# passed, the figure was wrong, and a human looked at pixels to say so.
# The corridor makes it mechanical — every CONTENT/ACCENT stroke owns a
# keep-out band of half its stroke width plus a pad, and label boxes must
# stay out. Two exemptions, both declarations rather than accidents:
# scaffolding roles (CONSTRUCTION/FRAME/MUTED) may be overwritten, and a
# label with halo=True is the author saying "this label rides busy ink" —
# the cartographic-halo idiom the corpus uses over curve families. What
# the gate polices is the unmarked case: a bare label on content ink is
# either a defect (move it — the nudge is computed) or a ride that must
# be declared (halo=True).

_CORRIDOR_ROLES = frozenset({Role.CONTENT, Role.ACCENT1, Role.ACCENT2})
_CORRIDOR_PAD = 2.0
# Sampling cap per curve: at spacing max(radius, 2px) this covers ~4000px of
# arc; longer curves sample coarser, erring by at most spacing/2.
_CORRIDOR_MAX_SAMPLES = 2048
# How far the gate searches for a free single-axis nudge before falling
# back to the free-region scan (larger than autoplace's budget on purpose:
# the gate may suggest a move the solver was not allowed to make).
_CORRIDOR_NUDGE_MAX = 36
_FREE_SCAN_STEP = 8.0


@dataclass(frozen=True)
class Corridor:
    samples: np.ndarray            # (M, 2) canvas px, spacing <= radius
    radius: float                  # stroke_width / 2 + pad
    what: str


def _corridor_samples(cpts: np.ndarray, spacing: float) -> np.ndarray:
    """Resample a canvas polyline at <= spacing along arc length (adaptive:
    capped at _CORRIDOR_MAX_SAMPLES for very long curves)."""
    seg = np.diff(cpts, axis=0)
    lens = np.hypot(seg[:, 0], seg[:, 1])
    total = float(lens.sum())
    if total == 0.0:
        return cpts[:1]
    n = int(min(max(total / spacing, 1), _CORRIDOR_MAX_SAMPLES))
    s = np.concatenate([[0.0], np.cumsum(lens)])
    u = np.linspace(0.0, total, n + 1)
    return np.column_stack([np.interp(u, s, cpts[:, 0]),
                            np.interp(u, s, cpts[:, 1])])


def ink_corridors(scene: Scene, style: Style, t: Transform,
                  offset: tuple[float, float] = (0.0, 0.0)) -> list[Corridor]:
    """Keep-out corridors for every content-role Curve/Vector, canvas px."""
    out: list[Corridor] = []
    for it in scene.items:
        if isinstance(it, Curve) and it.role in _CORRIDOR_ROLES:
            cpts = t.to_canvas_arr(np.asarray(it.pts, dtype=float))
            if it.closed:
                cpts = np.vstack([cpts, cpts[:1]])
        elif isinstance(it, Vector) and it.role in _CORRIDOR_ROLES:
            cpts = np.array([t.to_canvas(it.tail), t.to_canvas(it.tip)])
        else:
            continue
        w = style.ink(it.role).width * it.width_scale
        r = w / 2.0 + _CORRIDOR_PAD
        samples = _corridor_samples(cpts, max(r, 2.0))
        if offset != (0.0, 0.0):
            samples = samples + np.asarray(offset)
        out.append(Corridor(samples, r, f"{type(it).__name__} {it.role.name}"))
    return out


def corridor_hit(box: tuple[float, float, float, float],
                 corridors: list[Corridor]) -> Corridor | None:
    """The first corridor whose band intersects the box, else None."""
    for c in corridors:
        s, r = c.samples, c.radius
        if np.any((s[:, 0] >= box[0] - r) & (s[:, 0] <= box[2] + r)
                  & (s[:, 1] >= box[1] - r) & (s[:, 1] <= box[3] + r)):
            return c
    return None


def _shift(box: tuple[float, float, float, float], dx: float,
           dy: float) -> tuple[float, float, float, float]:
    return (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)


def _corridor_free_nudges(k: int, boxes: list[LabelBox],
                          corridors: list[Corridor], canvas_w: float,
                          canvas_h: float,
                          max_move: int = _CORRIDOR_NUDGE_MAX) -> list[tuple[float, float]]:
    """Smallest single-axis moves taking box k clear of every corridor,
    verified against all other boxes and the canvas. Smallest first, at
    most two."""
    box = boxes[k][2]
    out: list[tuple[float, float]] = []
    for m in range(1, max_move + 1):
        for dx, dy in ((float(m), 0.0), (-float(m), 0.0),
                       (0.0, float(m)), (0.0, -float(m))):
            cand = _shift(box, dx, dy)
            if cand[0] < 0 or cand[1] < 0 or cand[2] > canvas_w or cand[3] > canvas_h:
                continue
            if corridor_hit(cand, corridors) is not None:
                continue
            if any(_overlap(cand, boxes[j][2])
                   for j in range(len(boxes)) if j != k):
                continue
            out.append((dx, dy))
            if len(out) >= 2:
                return out
    return out


def _nearest_free_center(box: tuple[float, float, float, float],
                         other_boxes: list[tuple[float, float, float, float]],
                         corridors: list[Corridor], canvas_w: float,
                         canvas_h: float) -> tuple[float, float] | None:
    """Center of the nearest position where the box fits free of every
    corridor, other box, and the canvas edge — the fallback suggestion
    when no single-axis nudge frees the label."""
    w, h = box[2] - box[0], box[3] - box[1]
    if w > canvas_w or h > canvas_h:
        return None
    cx0, cy0 = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    best: tuple[float, float] | None = None
    best_d = math.inf
    for cx in np.arange(w / 2, canvas_w - w / 2 + 1e-9, _FREE_SCAN_STEP):
        for cy in np.arange(h / 2, canvas_h - h / 2 + 1e-9, _FREE_SCAN_STEP):
            d = math.hypot(cx - cx0, cy - cy0)
            if d >= best_d:
                continue
            cand = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            if corridor_hit(cand, corridors) is not None:
                continue
            if any(_overlap(cand, b) for b in other_boxes):
                continue
            best, best_d = (float(cx), float(cy)), d
    return best


def _label_ink_checks(entries: list[LabelEntry], corridors: list[Corridor],
                      style: Style, canvas_w: float, canvas_h: float,
                      to_math=None) -> list[Diagnostic]:
    """label-on-ink: a bare label box intersecting a content-ink corridor
    (halo=True declares the ride and exempts — see the section comment).
    The diagnostic carries the fix — verified free nudges, or the nearest
    ink-free region center when no single-axis nudge exists."""
    diags: list[Diagnostic] = []
    boxes = [box for box, _ in entries]
    for k, ((latex, _, box), owner) in enumerate(entries):
        if owner is not None and (getattr(owner, "halo", False)
                                  or (isinstance(owner, Callout) and owner.boxed)):
            continue
        hit = corridor_hit(box, corridors)
        if hit is None:
            continue
        nudges = _corridor_free_nudges(k, boxes, corridors, canvas_w, canvas_h)
        if nudges:
            opts = " or ".join(_fmt_nudge(d) for d in nudges)
            fix = f" — free: offset_px += {opts}, or declare halo=True"
        else:
            center = _nearest_free_center(box,
                                          [b[2] for i, b in enumerate(boxes) if i != k],
                                          corridors, canvas_w, canvas_h)
            if center is None:
                fix = " — no ink-free region fits this label; trim annotation"
            elif to_math is not None:
                mx, my = to_math(center)
                fix = (f" — no free single-axis nudge; nearest ink-free region "
                       f"center ({mx:.3g}, {my:.3g}) in math coords, or "
                       f"declare halo=True")
            else:
                fix = (f" — no free single-axis nudge; nearest ink-free region "
                       f"center ({center[0]:.0f}, {center[1]:.0f}) canvas px, "
                       f"or declare halo=True")
        diags.append(Diagnostic(
            "label-on-ink",
            f"'{latex}' sits on {hit.what} ink{fix}"))
    return diags


# arrows= fractions scanned for a clear replacement marker position
_ARROW_SCAN = np.linspace(0.05, 0.95, 19)


def _pt_box_gap(p: tuple[float, float],
                box: tuple[float, float, float, float]) -> float:
    dx = max(box[0] - p[0], p[0] - box[2], 0.0)
    dy = max(box[1] - p[1], p[1] - box[3], 0.0)
    return math.hypot(dx, dy)


def _arrow_mark_checks(scene: Scene, style: Style, t: Transform,
                       label_boxes: list[LabelBox],
                       offset: tuple[float, float] = (0.0, 0.0)) -> list[Diagnostic]:
    """arrow-on-mark: a direction marker on CONTENT ink landing on a Point
    or a label box. The diagnostic names a verified clear arrows= fraction.

    Only content-role curves are checked (the corridor role set): an
    ANNOTATION axis arrow ends at its label by design — the label names
    the axis at the tip — and scaffolding markers are not the defect."""
    from .render import _at_fraction

    ox, oy = offset
    pts_marks = [((t.to_canvas(p.xy)[0] + ox, t.to_canvas(p.xy)[1] + oy),
                  style.point_radius * p.radius_scale)
                 for p in scene.items if isinstance(p, Point)]
    diags: list[Diagnostic] = []
    for it in scene.items:
        if not (isinstance(it, Curve) and it.arrows
                and it.role in _CORRIDOR_ROLES):
            continue
        cpts = t.to_canvas_arr(np.asarray(it.pts, dtype=float))
        if it.closed:
            cpts = np.vstack([cpts, cpts[:1]])
        if offset != (0.0, 0.0):
            cpts = cpts + np.asarray(offset)
        head_len = style.arrowhead_len * it.arrow_scale

        def offender(frac: float) -> str | None:
            tip, _ = _at_fraction(cpts, frac)
            for (mx, my), pr in pts_marks:
                if math.hypot(tip[0] - mx, tip[1] - my) < pr + head_len:
                    return f"Point at canvas ({mx:.0f}, {my:.0f})"
            for latex, _, box in label_boxes:
                if _pt_box_gap(tip, box) < head_len:
                    return f"label '{latex}'"
            return None

        for frac in it.arrows:
            mark = offender(frac)
            if mark is None:
                continue
            clear = [u for u in _ARROW_SCAN if offender(float(u)) is None]
            if clear:
                best = min(clear, key=lambda u: abs(u - frac))
                fix = f" — clear: arrows=({best:.2f},)"
            else:
                fix = " — no clear fraction on this curve; drop the marker"
            diags.append(Diagnostic(
                "arrow-on-mark",
                f"arrowhead at t={frac:g} on Curve {it.role.name} sits on "
                f"{mark}{fix}"))
    return diags


def mechanical(scene: Scene, style: Style, width_px: float = 900) -> list[Diagnostic]:
    t = Transform(scene, width_px=width_px)
    entries = _label_entries(scene, style, t)
    boxes = [box for box, _ in entries]
    diags = _check_boxes(boxes, t.canvas_w, t.canvas_h,
                         type_scale=style.type_scale)
    corridors = ink_corridors(scene, style, t)
    diags += _label_ink_checks(entries, corridors, style, t.canvas_w,
                               t.canvas_h, to_math=t.from_canvas)
    diags += _arrow_mark_checks(scene, style, t, boxes)
    return diags


def _figure_label_entries(fig: "Figure", style: Style,
                          width_px: float) -> tuple[list[LabelEntry], float, float]:
    """Figure-wide entries in page coords: panel labels offset to their
    slots (owners kept — offset_px is canvas px in both frames), plus
    tags, connector labels, and page items. Tags and connector labels are
    derived geometry; page items are movable MathLabels."""
    from .figure import FRAME_INSET, TAG_PAD, connector_ink, layout_figure

    lay = layout_figure(fig, width_px)
    entries: list[LabelEntry] = []
    for panel, slot in zip(fig.panels, lay.slots):
        for (latex, pt, (x0, y0, x1, y1)), owner in _label_entries(
                panel.scene, style, slot.transform):
            entries.append(((latex, pt, (x0 + slot.x, y0 + slot.content_y,
                                         x1 + slot.x, y1 + slot.content_y)),
                            owner))
        if panel.tag:
            pt = style.label_size_pt
            entries.append(((panel.tag, pt, _box_at(
                panel.tag, pt, slot.x + FRAME_INSET + TAG_PAD,
                slot.y + FRAME_INSET + TAG_PAD, "left", "top")), None))
    for conn in fig.connectors:
        entries += [(_canvas_label_box(lab, style), None)
                    for lab in connector_ink(conn, lay, style).labels]
    entries += [(_canvas_label_box(lab, style), lab) for lab in fig.page_items]
    return entries, lay.canvas_w, lay.canvas_h


def figure_ink_corridors(fig: "Figure", style: Style,
                         width_px: float) -> list[Corridor]:
    """Every panel's content corridors, offset to figure canvas coords."""
    from .figure import layout_figure

    lay = layout_figure(fig, width_px)
    out: list[Corridor] = []
    for panel, slot in zip(fig.panels, lay.slots):
        out += ink_corridors(panel.scene, style, slot.transform,
                             offset=(slot.x, slot.content_y))
    return out


def mechanical_figure(fig: "Figure", style: Style, width_px: float = 900) -> list[Diagnostic]:
    """The same label checks run figure-wide: panel labels offset to page
    coords, plus tags, connector labels, and page items — collisions are
    checked across ALL of them, clipping against the figure canvas. Ink
    corridors and arrow marks are checked per panel in page coords; the
    free-region fallback reports canvas px (math coords are panel-local)."""
    from .figure import layout_figure

    entries, canvas_w, canvas_h = _figure_label_entries(fig, style, width_px)
    boxes = [box for box, _ in entries]
    diags = _check_boxes(boxes, canvas_w, canvas_h,
                         type_scale=style.type_scale)
    corridors = figure_ink_corridors(fig, style, width_px)
    diags += _label_ink_checks(entries, corridors, style, canvas_w, canvas_h)
    lay = layout_figure(fig, width_px)
    for panel, slot in zip(fig.panels, lay.slots):
        diags += _arrow_mark_checks(panel.scene, style, slot.transform, boxes,
                                    offset=(slot.x, slot.content_y))
    return diags


# --- the color gate ---------------------------------------------------------
#
# grammar.md already asserts what colour has to do — ink must be visible on
# paper, and hue must carry correspondence. Both are measurable, so neither is
# left to the eye. The mechanical gate rejects an 8 pt label as illegible; a
# stroke the reader cannot see is the same defect in a different channel.

# Content-bearing line work must clear this against the paper it lands on.
# WCAG's non-text floor: below it a stroke stops reading as a mark.
MIN_STROKE_CONTRAST = 3.0
# Scaffolding and un-outlined fills are *supposed* to recede — grammar.md's
# ink hierarchy makes construction and frame ink deliberately quiet, and
# holding them to the content floor would flatten the hierarchy that carries
# the figure's reading order. They only have to remain perceptible.
#
# Unlike the content floor this is not a standard; WCAG has nothing to say
# about ink that is meant to be quiet. It is anchored on the house style
# instead: the faintest deliberate ink in either theme is RISO's dotted frame
# at 1.44:1, so the floor sits just below that. The gate's job here is to
# catch ink that has vanished, not to second-guess a hairline.
MIN_PERCEPTIBLE_CONTRAST = 1.3

# Roles whose ink is scaffolding, not content.
_SCAFFOLD_ROLES = frozenset({Role.CONSTRUCTION, Role.FRAME, Role.MUTED})
# This many identical strokes is an ensemble, not a mark: grammar.md requires
# distributional claims to be *shown* (dense meshes, path bundles), and there
# the visible object is the bundle, whose ink accumulates where paths overlap.
# Holding one member to the content floor would forbid drawing the ensemble at
# all. The bundle still has to be perceptible, so it drops to the scaffold
# floor rather than out of the gate.
ENSEMBLE_MIN = 8
# Correspondence hues must stay this far apart under protanopia/deuteranopia.
MIN_CORRESPONDENCE_DELTA_E = 8.0
# A fill at or above this opacity IS the ground for marks anchored inside
# it: measuring them against the paper overstates their visibility.
HIGH_OPACITY_FILL = 0.6


def _stroke_color(it, style: Style) -> str | None:
    """The colour actually emitted for an item's line work (mirrors render)."""
    role_color = style.ink(it.role).color
    return getattr(it, "color", None) or role_color


def color_gate(scene: Scene, style: Style, width_px: float = 900,
               fill_grounds: bool = True) -> list[Diagnostic]:
    """Every mark visible on its ground; correspondence hues separable.

    Colours are composited against the paper before measurement — a stroke
    at opacity 0.3 is three times fainter than the hex it declares, and
    measuring the declared value overstates every faint mark in the corpus.

    A Point or MathLabel whose rendered position lies inside a high-opacity
    FilledCurve is measured against that fill (its actual ground), not the
    paper — the topmost containing fill wins. fill_grounds=False disables
    this (the pooled figure gate: panels share math coords, so cross-panel
    point-in-polygon would lie).
    """
    from .color import (compliant, composite, contrast, min_compliant_opacity,
                        worst_delta_e)
    from .geometry import point_in_poly
    from .scene import FilledCurve
    from .theme import CORRESPONDENCE, ORDER, SHADE

    papers = style.paper_stops() if hasattr(style, "paper_stops") else [style.background]
    diags: list[Diagnostic] = []
    # (color, alpha, floor) -> the mark wearing it, so one faint hue reports once
    seen: dict[tuple[str, float, float], str] = {}
    hues: dict[str, str] = {}

    def floor_for(color: str, role: Role, fill: bool) -> float | None:
        """None exempts the mark. A shading ramp is exempt by construction:
        its lit end is *meant* to approach the paper — relief comes from the
        step between neighbouring facets and from the surface edge, not from
        figure-ground contrast, and its readability is gated at the theme
        level as ramp monotonicity instead."""
        channel = getattr(color, "channel", None)
        if channel == SHADE:
            return None
        if fill:
            return MIN_PERCEPTIBLE_CONTRAST
        # An explicit correspondence/order hue is content whatever role it
        # rides on; otherwise the role decides.
        if channel in (CORRESPONDENCE, ORDER) or role not in _SCAFFOLD_ROLES:
            return MIN_STROKE_CONTRAST
        return MIN_PERCEPTIBLE_CONTRAST

    counts: dict[tuple[str, float], int] = {}

    def note(color: str, alpha: float, role: Role, what: str, fill: bool = False) -> None:
        floor = floor_for(color, role, fill)
        if floor is not None:
            key = (str(color), round(alpha, 2))
            counts[key] = counts.get(key, 0) + 1
            seen.setdefault(key + (floor,), what)

    def compliant_fix(color: str, paper: str, floor: float, alpha: float) -> str:
        """'nearest compliant: #xxxxxx (or opacity >= 0.yy)' — the repair,
        computed so the diagnostic is typed back, not searched for."""
        alt = compliant(str(color), paper, floor, opacity=alpha)
        if alt is None:
            return ""
        need = min_compliant_opacity(str(color), paper, floor) if alpha < 1.0 else None
        extra = f" (or opacity >= {need:.2f})" if need is not None else ""
        return f" — nearest compliant: {alt}{extra}"

    # High-opacity fills are local grounds: a Point/MathLabel rendered
    # inside one is read against the fill, not the paper.
    grounds: list[tuple[np.ndarray, tuple, str, float]] = [
        (np.asarray(it.pts, dtype=float), it.holes,
         str(it.color or style.ink(it.role).color), it.opacity)
        for it in scene.items
        if isinstance(it, FilledCurve) and it.pattern is None
        and it.opacity >= HIGH_OPACITY_FILL] if fill_grounds else []

    t: Transform | None = None

    def rendered_pos(it) -> tuple[float, float]:
        """Math coords of the mark as drawn: offset_px applied through the
        same transform render uses."""
        nonlocal t
        if isinstance(it, Point):
            return it.xy
        if it.offset_px == (0.0, 0.0):
            return it.anchor
        if t is None:
            t = Transform(scene, width_px=width_px)
        cx, cy = t.to_canvas(it.anchor)
        return t.from_canvas((cx + it.offset_px[0], cy + it.offset_px[1]))

    def containing_fill(pos) -> tuple[str, float] | None:
        """The topmost high-opacity fill containing pos, if any."""
        hit = None
        for pts, holes, color, alpha in grounds:
            if point_in_poly(pos, pts) and not any(
                    point_in_poly(pos, np.asarray(h)) for h in holes):
                hit = (color, alpha)
        return hit

    for it in scene.items:
        if isinstance(it, (Curve, Vector, Point, MathLabel, Brace, Callout)):
            c = _stroke_color(it, style)
            if not c:
                continue
            if getattr(c, "channel", None) == CORRESPONDENCE:
                hues[str(c)] = type(it).__name__
            over = None
            # A haloed label sits on its paper casing, not the fill under
            # it — the ordinary paper check is the honest one there.
            if grounds and isinstance(it, (Point, MathLabel)) \
                    and not getattr(it, "halo", False):
                over = containing_fill(rendered_pos(it))
            if over is not None:
                # Measured against the actual ground, not the paper.
                fill_c, fill_a = over
                floor = floor_for(c, it.role, False)
                if floor is None:
                    continue
                worst, worst_g = min(
                    (contrast(str(c), composite(fill_c, fill_a, p)),
                     composite(fill_c, fill_a, p)) for p in papers)
                if worst < floor:
                    diags.append(Diagnostic(
                        "faint-ink",
                        f"{type(it).__name__} {c} is {worst:.2f}:1 on fill "
                        f"{fill_c} it sits inside (floor {floor:.1f}:1) — the "
                        f"mark's ground is the fill, not the paper"
                        f"{compliant_fix(c, worst_g, floor, 1.0)}"))
                continue
            note(c, getattr(it, "opacity", 1.0), it.role, type(it).__name__)
        elif isinstance(it, FilledCurve):
            c = it.color or style.ink(it.role).color
            if c and not it.outline:
                note(c, it.opacity, it.role, "FilledCurve", fill=True)
            if it.edge_color:
                note(it.edge_color, 1.0, it.role, "FilledCurve edge")

    for (color, alpha, floor), what in seen.items():
        n = counts[(color, alpha)]
        if n >= ENSEMBLE_MIN:
            floor = min(floor, MIN_PERCEPTIBLE_CONTRAST)
        worst, worst_paper = min(
            ((contrast(composite(color, alpha, p), p), p) for p in papers))
        if worst < floor:
            alpha_note = f" at opacity {alpha:g}" if alpha < 1.0 else ""
            bundle = f" ({n} strokes — ensemble floor)" if n >= ENSEMBLE_MIN else ""
            diags.append(Diagnostic(
                "faint-ink",
                f"{what} {color}{alpha_note}{bundle} is {worst:.2f}:1 on paper "
                f"{worst_paper} (floor {floor:.1f}:1) — darken the ink or "
                f"lighten the paper, never leave the mark invisible"
                f"{compliant_fix(color, worst_paper, floor, alpha)}"))

    # Correspondence is all-pairs here: any two curves in a plane figure can
    # end up adjacent, unlike segments in a stack.
    cap = getattr(style, "correspondence_cap", None)
    palette = sorted(hues)
    if cap is not None and len(palette) > cap:
        diags.append(Diagnostic(
            "hue-collapse",
            f"{len(palette)} correspondence hues exceeds this theme's cap of "
            f"{cap} pairwise-distinct slots — encode identity as hue x dash, "
            f"facet into panels, or fold the tail"))
    for i in range(len(palette)):
        for j in range(i + 1, len(palette)):
            d, kind = worst_delta_e(palette[i], palette[j])
            if d < MIN_CORRESPONDENCE_DELTA_E:
                diags.append(Diagnostic(
                    "hue-collapse",
                    f"correspondence hues {palette[i]} and {palette[j]} are "
                    f"ΔE {d:.1f} apart under {kind} (floor "
                    f"{MIN_CORRESPONDENCE_DELTA_E:.0f}) — same hue, different "
                    f"object, to a reader who cannot separate them"))
    return diags


def color_gate_figure(fig: "Figure", style: Style) -> list[Diagnostic]:
    """The same checks across every panel of a multi-panel figure — hues are
    pooled, since correspondence is exactly what has to hold ACROSS panels.
    Fill-as-ground is skipped here: panels share math coordinates, so a
    pooled point-in-polygon test would read one panel's marks against
    another panel's fills."""
    from .scene import Scene
    pooled = Scene(items=[it for panel in fig.panels for it in panel.scene.items])
    return color_gate(pooled, style, fill_grounds=False)


def numerical(assert_fn: Callable[[], None]) -> list[Diagnostic]:
    try:
        assert_fn()
    except AssertionError as e:
        return [Diagnostic("numerical", str(e) or "assertion failed")]
    return []


class Checks:
    """Accumulating alternative to bare asserts inside assertions().

    Bare asserts stop at the first failure, so each gate run surfaces one
    defect; Checks collects them all and raises once:

        c = Checks()
        c.check(err < 1e-9, f"surface identity violated by {err:.2e}")
        c.check(d.min() < 0.12, "no spire at z = i")
        c.done()          # AssertionError listing every failure
    """

    def __init__(self) -> None:
        self.failures: list[str] = []

    def check(self, cond, msg: str) -> None:
        if not cond:
            self.failures.append(msg)

    def done(self) -> None:
        if self.failures:
            raise AssertionError("; ".join(self.failures))
