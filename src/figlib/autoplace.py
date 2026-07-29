"""Deterministic label placement: the free-nudge solver.

Promoted from the mechanical gate's diagnostics into a pre-gate pass —
the gate used to *compute* a verified collision-free nudge and print it
for someone to type back in; here the same computation is applied
directly, and gates then measure the solved scene (render sees the same
objects, so what is checked is what is drawn).

The solver only moves MathLabels (never brace/callout labels, whose
position is derived geometry), never past MAX_AUTO_NUDGE (beyond ~1.5
line heights a label detaches from its anchor — that is a design
problem, not a layout problem), and never a pinned label. Whatever it
cannot fix falls through to the gate exactly as before.
"""

from __future__ import annotations

from .gates import (Corridor, LabelEntry, _corridor_free_nudges, _entry_pad,
                    _figure_label_entries, _free_nudges, _inflate,
                    _label_entries, _NUDGE_PAD, _overlap, _shift,
                    corridor_hit, figure_ink_corridors, ink_corridors)
from .layout import Transform
from .scene import Scene
from .style import Style

# Largest offset the solver may apply, canvas px (single-axis for
# collisions, L1 for clip fixes). ~1.5 line heights at 11 pt.
MAX_AUTO_NUDGE = 24.0
# A move never creates a new overlap, so each pass only shrinks the
# colliding set; extra passes exist because one label moving can free a
# nudge for another that had none.
_MAX_PASSES = 4

Box = tuple[float, float, float, float]


def _clip_fix(box: Box, canvas_w: float, canvas_h: float) -> tuple[float, float]:
    """The translation pulling an out-of-frame box inside, with pad."""
    dx = dy = 0.0
    if box[0] < 0:
        dx = -box[0] + _NUDGE_PAD
    elif box[2] > canvas_w:
        dx = canvas_w - box[2] - _NUDGE_PAD
    if box[1] < 0:
        dy = -box[1] + _NUDGE_PAD
    elif box[3] > canvas_h:
        dy = canvas_h - box[3] - _NUDGE_PAD
    return dx, dy


def _resolve_one(k: int, boxes: list, canvas_w: float, canvas_h: float,
                 corridors: list[Corridor] = [],
                 pad: float = 0.0) -> tuple[float, float] | None:
    """The smallest verified free move for box k within budget, or None.

    Corridors (content-ink keep-out bands from the gate) are obstacles:
    a collision fix must not land on ink, and a label sitting on ink with
    no collision gets its own nudge search. pad is the halo inflation."""
    box = boxes[k][2]
    colliding = any(_overlap(box, boxes[m][2])
                    for m in range(len(boxes)) if m != k)
    if colliding:
        for dx, dy in _free_nudges(k, boxes, canvas_w, canvas_h):
            if abs(dx) + abs(dy) > MAX_AUTO_NUDGE:
                continue
            if corridor_hit(_inflate(_shift(box, dx, dy), pad), corridors) is not None:
                continue
            return (dx, dy)
        return None
    if corridors and corridor_hit(_inflate(box, pad), corridors) is not None:
        moves = _corridor_free_nudges(k, boxes, corridors, canvas_w, canvas_h,
                                      pad=pad, max_move=int(MAX_AUTO_NUDGE))
        return moves[0] if moves else None
    dx, dy = _clip_fix(box, canvas_w, canvas_h)
    if dx == 0.0 and dy == 0.0:
        return None
    if abs(dx) + abs(dy) > MAX_AUTO_NUDGE:
        return None
    cand = (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)
    if cand[0] < 0 or cand[1] < 0 or cand[2] > canvas_w or cand[3] > canvas_h:
        return None                      # bigger than the canvas
    if any(_overlap(cand, boxes[m][2]) for m in range(len(boxes)) if m != k):
        return None
    return (dx, dy)


def _solve(entries: list[LabelEntry], canvas_w: float, canvas_h: float,
           corridors: list[Corridor] = [], style: Style | None = None) -> list[str]:
    boxes = [box for box, _ in entries]
    notes: list[str] = []
    for _ in range(_MAX_PASSES):
        moved = False
        for k, (_, owner) in enumerate(entries):
            if owner is None or owner.pin:
                continue
            pad = _entry_pad(owner, style) if style is not None else 0.0
            delta = _resolve_one(k, boxes, canvas_w, canvas_h, corridors, pad)
            if delta is None:
                continue
            dx, dy = delta
            latex, pt, (x0, y0, x1, y1) = boxes[k]
            boxes[k] = (latex, pt, (x0 + dx, y0 + dy, x1 + dx, y1 + dy))
            owner.offset_px = (owner.offset_px[0] + dx, owner.offset_px[1] + dy)
            notes.append(f"'{latex}' offset_px += ({dx:+.0f}, {dy:+.0f})")
            moved = True
        if not moved:
            break
    return notes


def autoplace(scene: Scene, style: Style, width_px: float = 900) -> list[str]:
    """Nudge un-pinned MathLabels clear of collisions, content-ink
    corridors, and the canvas edge (in place); returns a note per move."""
    t = Transform(scene, width_px=width_px)
    return _solve(_label_entries(scene, style, t), t.canvas_w, t.canvas_h,
                  corridors=ink_corridors(scene, style, t), style=style)


def autoplace_figure(fig: "Figure", style: Style,
                     width_px: float = 900) -> list[str]:
    """The same pass figure-wide: panel labels and page items move; tags,
    connector labels, and content-ink corridors are fixed obstacles."""
    entries, canvas_w, canvas_h = _figure_label_entries(fig, style, width_px)
    return _solve(entries, canvas_w, canvas_h,
                  corridors=figure_ink_corridors(fig, style, width_px),
                  style=style)
