"""Deterministic gates: numerical assertions and mechanical render checks.

The readback gate (cold-reader test) lives in readback.py — it needs a
model; these two do not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from .layout import Transform
from .render import brace_ink, callout_ink
from .scene import Brace, Callout, MathLabel, Scene
from .style import Role, Style
from .typeset import render_math

# Fraction of a label's height assumed to hang below the baseline.
_DESCENT_FRAC = 0.2


@dataclass(frozen=True)
class Diagnostic:
    kind: str    # 'label-collision' | 'clipped' | 'tiny-label' | 'label-scale'
                 # | 'annotation-load' | 'numerical' | 'faint-ink' | 'hue-collapse'
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
            ha: str, va: str) -> tuple[float, float, float, float]:
    """Exact typeset bbox anchored at canvas (x, y)."""
    m = render_math(latex, size_pt)
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
    box = _box_at(it.latex, pt, x, y, it.ha, it.va)
    if it.angle_deg:
        box = _rotate_box(box, x, y, it.angle_deg)
    return (it.latex, pt, box)


def _label_boxes(scene: Scene, style: Style, t: Transform) -> list[LabelBox]:
    """Every box the mechanical gate owns: MathLabels (rotation-aware),
    Brace labels, Callout boxes — resolved through the same geometry the
    render emits."""
    boxes: list[LabelBox] = []
    for it in scene.items:
        if isinstance(it, MathLabel):
            pt = style.label_pt(it.size_pt)
            x, y = t.to_canvas(it.anchor)
            x += it.offset_px[0]
            y += it.offset_px[1]
            box = _box_at(it.latex, pt, x, y, it.ha, it.va)
            if it.angle_deg:
                box = _rotate_box(box, x, y, it.angle_deg)
            boxes.append((it.latex, pt, box))
        elif isinstance(it, Brace) and it.label is not None:
            _, lab = brace_ink(it, t, style)
            boxes.append(_canvas_label_box(lab, style))
        elif isinstance(it, Callout):
            pt = style.label_pt(it.size_pt)
            boxes.append((it.latex, pt, callout_ink(it, t, style).box))
    return boxes


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


def mechanical(scene: Scene, style: Style, width_px: float = 900) -> list[Diagnostic]:
    t = Transform(scene, width_px=width_px)
    return _check_boxes(_label_boxes(scene, style, t), t.canvas_w, t.canvas_h,
                        type_scale=style.type_scale)


def mechanical_figure(fig: "Figure", style: Style, width_px: float = 900) -> list[Diagnostic]:
    """The same label checks run figure-wide: panel labels offset to page
    coords, plus tags, connector labels, and page items — collisions are
    checked across ALL of them, clipping against the figure canvas."""
    from .figure import FRAME_INSET, TAG_PAD, connector_ink, layout_figure

    lay = layout_figure(fig, width_px)
    boxes: list[LabelBox] = []
    for panel, slot in zip(fig.panels, lay.slots):
        for latex, pt, (x0, y0, x1, y1) in _label_boxes(panel.scene, style, slot.transform):
            boxes.append((latex, pt, (x0 + slot.x, y0 + slot.content_y,
                                      x1 + slot.x, y1 + slot.content_y)))
        if panel.tag:
            pt = style.label_size_pt
            boxes.append((panel.tag, pt, _box_at(
                panel.tag, pt, slot.x + FRAME_INSET + TAG_PAD,
                slot.y + FRAME_INSET + TAG_PAD, "left", "top")))
    for conn in fig.connectors:
        boxes += [_canvas_label_box(lab, style)
                  for lab in connector_ink(conn, lay, style).labels]
    boxes += [_canvas_label_box(lab, style) for lab in fig.page_items]
    return _check_boxes(boxes, lay.canvas_w, lay.canvas_h,
                        type_scale=style.type_scale)


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


def _stroke_color(it, style: Style) -> str | None:
    """The colour actually emitted for an item's line work (mirrors render)."""
    role_color = style.ink(it.role).color
    return getattr(it, "color", None) or role_color


def color_gate(scene: Scene, style: Style) -> list[Diagnostic]:
    """Every mark visible on its ground; correspondence hues separable.

    Colours are composited against the paper before measurement — a stroke
    at opacity 0.3 is three times fainter than the hex it declares, and
    measuring the declared value overstates every faint mark in the corpus.
    """
    from .color import composite, contrast, worst_delta_e
    from .scene import (Brace, Callout, Curve, FilledCurve, MathLabel, Point,
                        Vector)
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

    for it in scene.items:
        if isinstance(it, (Curve, Vector, Point, MathLabel, Brace, Callout)):
            c = _stroke_color(it, style)
            if c:
                note(c, getattr(it, "opacity", 1.0), it.role, type(it).__name__)
                if getattr(c, "channel", None) == CORRESPONDENCE:
                    hues[str(c)] = type(it).__name__
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
                f"lighten the paper, never leave the mark invisible"))

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
    pooled, since correspondence is exactly what has to hold ACROSS panels."""
    from .scene import Scene
    pooled = Scene(items=[it for panel in fig.panels for it in panel.scene.items])
    return color_gate(pooled, style)


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
