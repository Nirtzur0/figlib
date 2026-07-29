"""Multi-panel figures: Needham's page grammar as a container.

A Figure holds Panels (each a Scene with its own math->canvas Transform),
Connectors between them (squiggle map-arrows, block arrows, =/+ glyphs),
and a page layer of MathLabels anchored in figure canvas px.

Layout is dumb affine placement — grid slots plus gaps, no constraint
solver; the mechanical gate catches whatever collides. Every glyph is an
explicit polyline/polygon in canvas px, so all ink has a bbox we own.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .layout import Transform
from .scene import MathLabel, Scene
from .style import Style


@dataclass
class Panel:
    scene: Scene
    tag: str | None = None            # "[a]", typeset inside the top-left corner
    frame: bool = True                # rounded dotted border at the slot bounds
    width_frac: float | None = None   # fraction of the row's post-gap width


@dataclass
class Connector:
    src: int                          # panel indices
    dst: int
    kind: str = "map"                 # map | block | eq | plus
    label: str | None = None          # typeset above the glyph
    label_below: str | None = None    # typeset below it
    double: bool = False              # second reversed squiggle beneath (forward/back maps)


@dataclass
class Figure:
    panels: list[Panel]
    connectors: list[Connector] = field(default_factory=list)
    grid: tuple[int, int] | None = None   # (rows, cols), row-major fill; None -> one row
    gap_px: float = 46.0
    # page layer: MathLabels anchored in figure canvas px, drawn last
    page_items: list[MathLabel] = field(default_factory=list)


# Frame furniture (canvas px).
FRAME_INSET = 2.0
FRAME_RADIUS = 10.0
TAG_PAD = 10.0

# Map-squiggle shape (canvas px).
SQUIGGLE_WIGGLES = 2.5
SQUIGGLE_AMP = 3.5
GLYPH_PAD = 5.0          # clearance between glyph and each panel edge
DOUBLE_OFFSET = 9.0      # the reversed squiggle sits this far beneath
OPERATOR_PT = 17.0       # size of the =/+ glyphs between panels


@dataclass
class Slot:
    x: float                 # frame bounds; h is the full row height
    y: float
    w: float
    h: float
    content_y: float         # top of the panel's own canvas (vertically centered)
    transform: Transform


@dataclass
class FigureLayout:
    slots: list[Slot]
    canvas_w: float
    canvas_h: float


def layout_figure(fig: Figure, width_px: float) -> FigureLayout:
    """Slot geometry: row-major grid fill, widths split minus gaps
    (width_frac honored), each panel at its slot width through its own
    Transform; row height = max panel height, panels vertically centered."""
    rows, cols = fig.grid or (1, len(fig.panels))
    slots: list[Slot] = []
    y = 0.0
    for r in range(rows):
        row = fig.panels[r * cols:(r + 1) * cols]
        if not row:
            break
        avail = width_px - (len(row) - 1) * fig.gap_px
        spec = sum(p.width_frac for p in row if p.width_frac is not None)
        n_free = sum(1 for p in row if p.width_frac is None)
        free_w = avail * max(1.0 - spec, 0.0) / n_free if n_free else 0.0
        widths = [avail * p.width_frac if p.width_frac is not None else free_w
                  for p in row]
        transforms = [Transform(p.scene, width_px=w) for p, w in zip(row, widths)]
        row_h = max(t.canvas_h for t in transforms)
        x = 0.0
        for w, t in zip(widths, transforms):
            slots.append(Slot(x=x, y=y, w=w, h=row_h,
                              content_y=y + (row_h - t.canvas_h) / 2, transform=t))
            x += w + fig.gap_px
        y += row_h + fig.gap_px
    return FigureLayout(slots=slots, canvas_w=width_px, canvas_h=y - fig.gap_px)


@dataclass
class ConnectorInk:
    """Connector geometry resolved to canvas px — shared by render (draws
    it) and the mechanical gate (boxes the labels)."""
    strokes: list[np.ndarray]        # open polylines, ANNOTATION ink
    heads: list[np.ndarray]          # filled arrowhead triangles (3, 2)
    hollow: list[np.ndarray]         # paper-filled outlined polygons (block arrow)
    labels: list[MathLabel]          # anchors in canvas px


def _endpoints(a: Slot, b: Slot) -> tuple[np.ndarray, np.ndarray]:
    """Facing edge midpoints along the dominant axis between slot centers."""
    ca = np.array([a.x + a.w / 2, a.y + a.h / 2])
    cb = np.array([b.x + b.w / 2, b.y + b.h / 2])
    d = cb - ca
    if abs(d[0]) >= abs(d[1]):
        ymid = (ca[1] + cb[1]) / 2
        if d[0] >= 0:
            return np.array([a.x + a.w, ymid]), np.array([b.x, ymid])
        return np.array([a.x, ymid]), np.array([b.x + b.w, ymid])
    xmid = (ca[0] + cb[0]) / 2
    if d[1] >= 0:
        return np.array([xmid, a.y + a.h]), np.array([xmid, b.y])
    return np.array([xmid, a.y]), np.array([xmid, b.y + b.h])


def _squiggle(start: np.ndarray, end: np.ndarray, style: Style) -> tuple[np.ndarray, np.ndarray]:
    """Sine-wiggle polyline from start toward end plus a filled head at end.
    Whole wiggle count, so the wave leaves and lands on the axis."""
    from .render import _arrowhead  # render imports this module; lazy breaks the cycle

    v = end - start
    n = float(np.hypot(*v)) or 1.0
    u = v / n
    perp = np.array([-u[1], u[0]])
    body = max(n - 0.7 * style.arrowhead_len, 1.0)
    s = np.linspace(0.0, body, 48)
    wave = SQUIGGLE_AMP * np.sin(2 * np.pi * SQUIGGLE_WIGGLES * s / body)
    pts = start + np.outer(s, u) + np.outer(wave, perp)
    head = np.array(_arrowhead((float(end[0]), float(end[1])),
                               (float(u[0]), float(u[1])), style, scale=0.9))
    return pts, head


def _block_arrow(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Hollow block arrow polygon (filmstrip step)."""
    v = end - start
    n = float(np.hypot(*v)) or 1.0
    u = v / n
    perp = np.array([-u[1], u[0]])
    hb, hh, hl = 5.0, 10.0, 13.0        # body half, head half, head length
    neck = start + max(n - hl, 1.0) * u
    return np.array([
        start - hb * perp, neck - hb * perp, neck - hh * perp, end,
        neck + hh * perp, neck + hb * perp, start + hb * perp,
    ])


def connector_ink(conn: Connector, lay: FigureLayout, style: Style) -> ConnectorInk:
    p0, p1 = _endpoints(lay.slots[conn.src], lay.slots[conn.dst])
    v = p1 - p0
    n = float(np.hypot(*v)) or 1.0
    u = v / n
    perp = np.array([-u[1], u[0]])
    a = p0 + GLYPH_PAD * u
    b = p1 - GLYPH_PAD * u
    mid = (p0 + p1) / 2
    ink = ConnectorInk([], [], [], [])

    half_above = SQUIGGLE_AMP
    half_below = SQUIGGLE_AMP
    if conn.kind == "map":
        pts, head = _squiggle(a, b, style)
        ink.strokes.append(pts)
        ink.heads.append(head)
        if conn.double:
            pts2, head2 = _squiggle(b + DOUBLE_OFFSET * perp, a + DOUBLE_OFFSET * perp, style)
            ink.strokes.append(pts2)
            ink.heads.append(head2)
            half_below += DOUBLE_OFFSET
    elif conn.kind == "block":
        ink.hollow.append(_block_arrow(a, b))
        half_above = half_below = 10.0
    elif conn.kind in ("eq", "plus"):
        glyph = "=" if conn.kind == "eq" else "+"
        ink.labels.append(MathLabel(glyph, (float(mid[0]), float(mid[1])),
                                    ha="center", va="center", size_pt=OPERATOR_PT))
        half_above = half_below = OPERATOR_PT * 0.75
    else:
        raise ValueError(f"unknown connector kind {conn.kind!r}")

    if conn.label:
        ink.labels.append(MathLabel(conn.label, (float(mid[0]), float(mid[1])),
                                    ha="center", va="bottom",
                                    offset_px=(0.0, -(half_above + 6.0))))
    if conn.label_below:
        ink.labels.append(MathLabel(conn.label_below, (float(mid[0]), float(mid[1])),
                                    ha="center", va="top",
                                    offset_px=(0.0, half_below + 6.0)))
    return ink
