"""Placement along an author-chosen locus, and the clearance query.

`autoplace` closes one hole: an un-pinned label that collides gets a small
verified `offset_px` nudge. The transcripts show the expensive hole is the
other one — a label whose position IS a parameter of the geometry. Text set
along an arc, a tag at a fraction of a curve, a legend on a ring: those are
pinned by construction (their placement carries meaning), `autoplace` is
forbidden to touch them, and so they got hand-searched instead. One figure
took four render cycles to settle a single radius (1.17 -> 1.32 -> 1.40 ->
1.52), with the clearance arithmetic done by hand in prose each round.

The split this module draws is the one the doctrine already asks for
(primitive-gaps.md: model proposes, gates dispose):

* **The author chooses the LOCUS.** That is the meaning-bearing decision —
  "this word belongs on the rim of the sphere", "this tag rides the
  separatrix" — and no search can recover it.
* **The library chooses the POINT on it.** That is a 1-D scan over
  candidates the program already stated, with one answer and no objective
  over layouts. Same species as `boundary_toward` or a longest-path rank:
  derived, not solved.
* **The achieved clearance comes back as a number**, so the program
  asserts on it. A placement that silently degrades when the figure
  changes is exactly what the hand-tuned constant did.

Nothing here moves an item. `place_on_locus` returns a `Placement`; the
figure program builds the label from it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .gates import (Corridor, _box_at, _label_boxes, _label_entries,
                    _rotate_box, ink_corridors)
from .layout import Transform
from .scene import MathLabel, Scene
from .style import Style

XY = tuple[float, float]
Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Placement:
    """Where a label goes, and how much room it got."""

    anchor: XY               # math coords: a point OF the locus, never between
    index: int               # which candidate won
    clearance_px: float      # gap to the nearest content ink or existing label
    angle_deg: float = 0.0   # locus tangent on the page, if asked for

    def label(self, latex: str, **kw) -> MathLabel:
        """The pinned `MathLabel` this placement describes. Pinned because
        the position is now derived from the geometry — `autoplace` nudging
        it afterwards would undo the solve."""
        kw.setdefault("ha", "center")
        kw.setdefault("va", "center")
        return MathLabel(latex, self.anchor, angle_deg=self.angle_deg,
                         pin=True, **kw)


def _box_gap(a: Box, b: Box) -> float:
    """Separation between two canvas boxes; 0 when they touch or overlap."""
    dx = max(a[0] - b[2], b[0] - a[2], 0.0)
    dy = max(a[1] - b[3], b[1] - a[3], 0.0)
    return math.hypot(dx, dy)


def _corridor_gap(box: Box, corridors: Sequence[Corridor]) -> float:
    """Smallest gap from `box` to any ink corridor's BAND (samples inflated
    by the corridor radius). Negative when the box is on the ink, which is
    what makes the returned clearance usable as an assertion target."""
    best = math.inf
    for c in corridors:
        s = c.samples
        dx = np.maximum(np.maximum(box[0] - s[:, 0], s[:, 0] - box[2]), 0.0)
        dy = np.maximum(np.maximum(box[1] - s[:, 1], s[:, 1] - box[3]), 0.0)
        best = min(best, float(np.min(np.hypot(dx, dy))) - c.radius)
    return best


def _clearance(box: Box, corridors: Sequence[Corridor],
               boxes: Sequence[Box]) -> float:
    gaps = [_corridor_gap(box, corridors)]
    gaps += [_box_gap(box, b) for b in boxes]
    return min(gaps) if gaps else math.inf


def _readable(angle_deg: float) -> float:
    """Fold a tangent angle into (-90, 90] so text never sets upside down."""
    a = (angle_deg + 180.0) % 360.0 - 180.0
    if a > 90.0:
        a -= 180.0
    elif a <= -90.0:
        a += 180.0
    return a


def _tangent_deg(canvas: np.ndarray, i: int) -> float:
    """Page angle of the locus at candidate i, CCW (canvas +y is DOWN, so
    the sign flips relative to canvas coordinates)."""
    if len(canvas) < 2:
        return 0.0
    j = min(i + 1, len(canvas) - 1)
    k = max(j - 1, 0)
    d = canvas[j] - canvas[k]
    if not np.any(d):
        return 0.0
    return _readable(math.degrees(math.atan2(-d[1], d[0])))


def _label_box(latex: str, pt: float, cx: float, cy: float, ha: str, va: str,
               angle_deg: float) -> Box:
    box = _box_at(latex, pt, cx, cy, ha, va)
    return _rotate_box(box, cx, cy, angle_deg) if angle_deg else box


def place_on_locus(scene: Scene, style: Style, latex: str,
                   locus: Sequence[XY], *, width_px: float = 900.0,
                   size_pt: float | None = None, ha: str = "center",
                   va: str = "center", angle_from_tangent: bool = False,
                   angles: Sequence[float] | None = None,
                   ) -> Placement:
    """The candidate on `locus` with the most room, measured exactly.

    `locus` is a sequence of math-coord points the author is willing to put
    the label on — sample it as finely as the choice deserves. Obstacles
    are the same ones the mechanical gate uses: content-ink corridors and
    every label already in the scene. Ties break to the lowest index, so
    the result is deterministic and a re-run never silently reshuffles.

    Rotation, if the label needs it, comes from one of two places:

    * `angle_from_tangent` differences neighbouring candidates, which is
      only meaningful when the locus is a PATH (an arc, a curve). A locus
      that sweeps two parameters is a point cloud, and consecutive entries
      are not neighbours — ask for tangents there and you get noise.
    * `angles`, one per candidate, when the caller knows the true
      orientation. A word set in a 3-D plane knows its tangent from the
      projection; the library would only be guessing it back off the page.

    The scene is not modified; build the label from the returned
    `Placement` (or `Placement.label`).
    """
    pts = [(float(p[0]), float(p[1])) for p in locus]
    if not pts:
        raise ValueError("place_on_locus needs at least one candidate point")
    if angles is not None:
        if angle_from_tangent:
            raise ValueError("give angles= or angle_from_tangent, not both")
        if len(angles) != len(pts):
            raise ValueError(f"angles has {len(angles)} entries for "
                             f"{len(pts)} candidates")
    t = Transform(scene, width_px=width_px)
    corridors = ink_corridors(scene, style, t)
    boxes = [b for _latex, _pt, b in _label_boxes(scene, style, t)]
    pt_size = style.label_pt(size_pt)
    canvas = t.to_canvas_arr(np.asarray(pts, dtype=float))

    best_i, best_c, best_a = 0, -math.inf, 0.0
    for i, (cx, cy) in enumerate(canvas):
        if angles is not None:
            ang = _readable(float(angles[i]))
        else:
            ang = _tangent_deg(canvas, i) if angle_from_tangent else 0.0
        box = _label_box(latex, pt_size, float(cx), float(cy), ha, va, ang)
        c = _clearance(box, corridors, boxes)
        if c > best_c:                      # strict: ties keep the lower index
            best_i, best_c, best_a = i, c, ang
    return Placement(pts[best_i], best_i, float(best_c), float(best_a))


def clearances(scene: Scene, style: Style, *, width_px: float = 900.0
               ) -> list[tuple[float, str]]:
    """(clearance_px, latex) for every label in the scene, tightest first.

    One pass: corridors and label boxes are built once, unlike calling
    `label_clearance` per label.
    """
    t = Transform(scene, width_px=width_px)
    corridors = ink_corridors(scene, style, t)
    entries = _label_entries(scene, style, t)
    boxes = [b for (_l, _p, b), _o in entries]
    out = [(_clearance(box, corridors, boxes[:i] + boxes[i + 1:]), latex)
           for i, ((latex, _pt, box), _o) in enumerate(entries)]
    out.sort(key=lambda kv: kv[0])
    return out


def clearance_signal(scene, style, width_px: float = 900.0, top: int = 3
                     ) -> list[str]:
    """Advisory line for `Report.signals`: the tightest labels on the page.

    212 PNG looks across the corpus against 164 edits, and half of those
    looks were followed by ANOTHER look — the loop was largely asking
    "does this label touch that stroke?", a question with a number for an
    answer. `--report` answers it and was used in 4% of runs, so this rides
    the channel that runs every time instead of the one behind a flag.

    Advisory only. The mechanical gate still owns pass/fail; a tight label
    that clears is legal and sometimes correct.
    """
    if not isinstance(scene, Scene):
        return []                                # Figure: panel-local coords
    got = clearances(scene, style, width_px=width_px)
    if not got:
        return []
    shown = ", ".join(f"{lat!r} {c:.1f}px" for c, lat in got[:top])
    return [f"clearance: tightest labels {shown}"]


def label_clearance(scene: Scene, style: Style, label: MathLabel, *,
                    width_px: float = 900.0) -> float:
    """Room around a label already in the scene, canvas px; <= 0 means it
    is on ink or on another label.

    The query that replaces doing this arithmetic by hand in a reasoning
    block. Its own box is excluded, so a label never blocks itself.
    """
    t = Transform(scene, width_px=width_px)
    corridors = ink_corridors(scene, style, t)
    # entries carry their owner, so identity picks out this label's own box.
    # Zipping against scene.items would desync on the first Brace or Callout,
    # whose boxes are derived geometry with no MathLabel of their own.
    own = None
    boxes: list[Box] = []
    for (_latex, _pt, b), owner in _label_entries(scene, style, t):
        if owner is label:
            own = b
        else:
            boxes.append(b)
    if own is None:
        raise ValueError("label is not in this scene")
    return _clearance(own, corridors, boxes)
