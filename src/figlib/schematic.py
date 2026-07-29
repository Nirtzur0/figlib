"""Class B: schematics as typed nodes, ports, and edges.

A producer module in the surface3d/builders mold — dataclasses and pure
functions that emit ordinary scene Items, never a renderer. For a Class B
figure `compute()` carries no numerics and **layout is the content**: the
program states every coordinate and the gates dispose. Nothing here routes
around an obstacle, packs, or minimizes crossings — the layout pass in the
second half of this file DERIVES three things (a boundary intersection, a
box that holds its own label, a longest-path rank) and every one of them
can be replaced by a stated value that wins locally. The checks at the
bottom of the first half are how a wrong layout gets caught.

The three commitments:

* **Ports, not centers.** An edge attaches at a named boundary anchor
  (`"left"`, `"top@0.25"`, `("right", 0.8)`), so fan-in and fan-out read
  as separate arrivals instead of a pencil of lines through a centroid.
* **Edge kind routes to a decoration policy** (`EDGE_KINDS`), expressed
  entirely in existing `Curve` fields — dash, arrowhead fill, head count,
  a flat terminal bar. The theme is untouched; kind is meaning, not paint.
* **Routing is chosen, never solved.** Straight, one axis-aligned elbow at
  a corner you give, or a Bezier that *passes through* the via point(s)
  you give. If an edge collides with a box, that is a layout decision to
  revise, and `clearance_violations` reports it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .scene import Curve, FilledCurve, Item, MathLabel
from .style import Role
from .typeset import render_math

XY = tuple[float, float]
# A port spec is "left" | "top@0.25" | ("right", 0.8).
_SIDES = ("left", "right", "bottom", "top")

# Two terminal glyphs are drawn geometry rather than renderer-sized marks,
# so their lengths are in MATH units and a schematic states them the way it
# states every other coordinate. Override per edge.
BAR_HALF = 0.12       # half-length of an `inhibit` edge's flat terminal bar
CHEVRON_GAP = 0.15    # spacing between the stacked heads of a `copy` edge
TRUNC_DIAMOND_HALF = 0.10   # half-diagonal of a truncated edge's terminator

# The elision stack's per-card offset, as a FRACTION of min(w, h) rather
# than a math length: a stack has to read as the same lie at every box
# size, and a fixed offset would swallow a small node and vanish on a big
# one. Dimensionless, so it is the one constant here not in math units.
STACK_OFFSET = 0.10


# --- geometry ---------------------------------------------------------------


def rounded_rect(center: XY, width: float, height: float, radius: float = 0.0,
                 n_arc: int = 6) -> np.ndarray:
    """Closed rounded-rectangle polygon, CCW from the bottom-right corner.

    `radius` is clamped to half the shorter side, so radius >= h/2 gives a
    pill — the only shape variant a rails-and-paths schematic has needed.
    """
    cx, cy = float(center[0]), float(center[1])
    w, h = abs(float(width)), abs(float(height))
    r = float(min(max(radius, 0.0), 0.5 * min(w, h)))
    x0, x1 = cx - w / 2.0, cx + w / 2.0
    y0, y1 = cy - h / 2.0, cy + h / 2.0
    if r == 0.0:
        return np.array([[x1, y0], [x1, y1], [x0, y1], [x0, y0]])
    th = np.linspace(0.0, np.pi / 2.0, max(int(n_arc), 2))
    corners = (((x1 - r, y0 + r), -np.pi / 2.0), ((x1 - r, y1 - r), 0.0),
               ((x0 + r, y1 - r), np.pi / 2.0), ((x0 + r, y0 + r), np.pi))
    return np.vstack([
        np.column_stack([ccx + r * np.cos(phase + th), ccy + r * np.sin(phase + th)])
        for (ccx, ccy), phase in corners])


def _parse_port(spec) -> tuple[str, float]:
    """`"left"` | `"top@0.25"` | `("right", 0.8)` -> (side, t in [0, 1])."""
    if isinstance(spec, (tuple, list)):
        side, t = spec[0], float(spec[1])
    elif "@" in spec:
        side, _, ts = spec.partition("@")
        t = float(ts)
    else:
        side, t = spec, 0.5
    side = side.strip().lower()
    if side not in _SIDES:
        raise ValueError(f"port side {side!r} not one of {_SIDES}")
    return side, t


def _as_xy(p) -> np.ndarray:
    return np.asarray(p, dtype=float).reshape(2)


def _arclen(pts: np.ndarray) -> np.ndarray:
    seg = np.diff(pts, axis=0)
    return np.concatenate([[0.0], np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))])


def point_at(pts: np.ndarray, t: float) -> XY:
    """Point at arc-length fraction t of a polyline (render's convention)."""
    pts = np.asarray(pts, dtype=float)
    s = _arclen(pts)
    total = float(s[-1])
    if total <= 0.0:
        return (float(pts[0, 0]), float(pts[0, 1]))
    target = min(max(float(t), 0.0), 1.0) * total
    i = int(np.clip(np.searchsorted(s, target) - 1, 0, len(pts) - 2))
    ds = s[i + 1] - s[i]
    f = (target - s[i]) / ds if ds > 0 else 0.0
    p = pts[i] + f * (pts[i + 1] - pts[i])
    return (float(p[0]), float(p[1]))


# --- nodes ------------------------------------------------------------------


class _Box:
    """Rect geometry shared by every axis-aligned box: `Node` and `Region`.

    Both are a center plus a width, a height and a corner radius, and both
    need the same ports, containment and boundary distance. What separates
    them is what they MEAN — a `Node` is a thing in the graph and an edge
    that enters one it is not attached to is a collision; a `Region` is a
    ground and edges cross it freely — so they stay distinct types and only
    the geometry is shared.
    """

    center: XY
    width: float
    height: float
    n_arc: int

    @property
    def radius(self) -> float:
        raise NotImplementedError

    @property
    def rect(self) -> tuple[float, float, float, float]:
        """(x0, y0, x1, y1)."""
        cx, cy = self.center
        return (cx - self.width / 2.0, cy - self.height / 2.0,
                cx + self.width / 2.0, cy + self.height / 2.0)

    def outline(self) -> np.ndarray:
        return rounded_rect(self.center, self.width, self.height, self.radius,
                            n_arc=self.n_arc)

    def port(self, spec="top") -> XY:
        """Boundary anchor. t runs 0 -> 1 with increasing x on top/bottom and
        with increasing y on left/right, so t = 0.5 is always the midpoint."""
        side, t = _parse_port(spec)
        x0, y0, x1, y1 = self.rect
        if side == "left":
            return (x0, y0 + t * (y1 - y0))
        if side == "right":
            return (x1, y0 + t * (y1 - y0))
        if side == "bottom":
            return (x0 + t * (x1 - x0), y0)
        return (x0 + t * (x1 - x0), y1)

    def contains(self, p, pad: float = 0.0) -> bool:
        x0, y0, x1, y1 = self.rect
        px, py = float(p[0]), float(p[1])
        return (x0 - pad <= px <= x1 + pad) and (y0 - pad <= py <= y1 + pad)

    def boundary_distance(self, p) -> float:
        """Unsigned distance from p to the box's RECT boundary.

        The rect, not the rounded outline: rounding is cosmetic, and a port
        near a corner would otherwise read as detached from its own box.
        """
        x0, y0, x1, y1 = self.rect
        px, py = float(p[0]), float(p[1])
        ox = max(x0 - px, 0.0, px - x1)
        oy = max(y0 - py, 0.0, py - y1)
        if ox > 0.0 or oy > 0.0:
            return float(np.hypot(ox, oy))
        return float(min(px - x0, x1 - px, py - y0, y1 - py))


@dataclass(frozen=True)
class Node(_Box):
    """A box at an explicit position and size, with a centered label.

    Size is a stated number, never a growing box: a `Node` that resized
    itself would hide overflow instead of reporting it. `auto_node` will
    COMPUTE that number from the label's typeset metrics if the program
    asks, but the value still lands in this field and `label_overflow`
    still checks it. The mechanical gate catches label collisions.

    `fill` is a paper colour (`THEME.background` / `THEME.paper[0]`), which
    is what lets rails and edges pass cleanly BEHIND a node.

    Two of the fields are honesty marks — elisions drawn instead of
    confessed in a docstring. `stack=n` puts n ghost cards behind the box:
    "this node is many things abbreviated to one", which is what a box
    labelled *Multi-Head Attention* already means and currently only says
    in prose. `dash` makes the outline provisional; `unknown_node` is the
    named case, a box whose mechanism nobody knows.
    """

    key: str
    center: XY
    width: float
    height: float
    label: str | None = None
    role: Role = Role.CONTENT
    corner: float | None = None        # radius in math units; None -> 0.22*min(w,h)
    fill: str | None = None            # paper colour, or None for an open box
    label_role: Role | None = None     # None -> the node's own role
    label_size_pt: float | None = None
    label_offset_px: XY = (0.0, 0.0)
    width_scale: float = 1.0
    n_arc: int = 6
    stack: int = 0                     # ghost cards behind the box: "one of many"
    dash: str | None = None            # provisional outline; see `unknown_node`

    @property
    def radius(self) -> float:
        r = self.corner if self.corner is not None else 0.22 * min(self.width, self.height)
        return float(min(max(r, 0.0), 0.5 * min(self.width, self.height)))

    def _card(self, pts: np.ndarray, role: Role) -> list[Item]:
        """One box body at `pts`: a wash, an outline, or a wash under one.

        `FilledCurve` has no dash channel — dash is a stroke property and
        the fill's outline is drawn by the fill — so a dashed filled box is
        the one case that costs two items: an outline-free wash with its
        own dashed stroke on top of it.
        """
        if self.fill is None:
            return [Curve(pts, role=role, closed=True, dash=self.dash,
                          width_scale=self.width_scale)]
        out: list[Item] = [FilledCurve(pts, role=role, color=self.fill,
                                       opacity=1.0, outline=self.dash is None)]
        if self.dash is not None:
            out.append(Curve(pts, role=role, closed=True, dash=self.dash,
                             width_scale=self.width_scale))
        return out

    def items(self) -> list[Item]:
        out: list[Item] = []
        pts = self.outline()
        # Ghosts furthest-first, so the stack reads front-to-back and the
        # body paints over all of them. MUTED because a ghost is the same
        # object off-focus — grammar.md's off-focus channel exactly.
        d0 = STACK_OFFSET * min(abs(self.width), abs(self.height))
        for k in range(int(self.stack), 0, -1):
            out += self._card(pts + [k * d0, -k * d0], Role.MUTED)
        out += self._card(pts, self.role)
        if self.label:
            out.append(MathLabel(self.label, self.center,
                                 role=self.label_role or self.role,
                                 ha="center", va="center",
                                 size_pt=self.label_size_pt,
                                 offset_px=self.label_offset_px))
        return out


# `?\,?\,?` and not `???`: three question marks set as math run together
# into one blob, and the thin spaces are what make it read as a placeholder
# for three unknown things rather than punctuation.
UNKNOWN_LABEL = r"?\,?\,?"


def unknown_node(key: str, center: XY, width: float, height: float, **kw) -> Node:
    """A box whose mechanism is not known — dashed outline, `???` inside.

    Uncertainty as a node rather than a caption hedge. It is two settings,
    but naming it is the point: a figure that draws this has committed to
    where its ignorance lives, and a reader can see the gap without reading
    the caption.
    """
    kw.setdefault("label", UNKNOWN_LABEL)
    kw.setdefault("dash", "dashed")
    return Node(key, center, width, height, **kw)


# --- regions: grouping as a ground, not a line ------------------------------
#
# The transformer-circuits idiom. A group of nodes gets a pale filled
# rounded rect BEHIND it, and nesting is read off a contrast ladder —
# paper, then region, then node fill — with no line drawn anywhere. It is
# the cheapest grouping channel there is: it costs no ink the reader has to
# decode, and it composes, because a region inside a region is just a
# second step up the ladder.
#
# The corpus washes its groups at about 1.03:1 against white. The house
# floor for a fill is MIN_PERCEPTIBLE_CONTRAST (1.3:1), and on paper with a
# grain texture that floor is right and the corpus value would vanish, so
# REGION_OPACITY is pinned AT the floor rather than at the reference. A
# region that wants to be fainter has to carry an outline instead, which is
# a different (and louder) grouping channel — the colour gate exempts an
# outlined fill because the line, not the wash, is then doing the work.
REGION_OPACITY = 0.28


@dataclass(frozen=True)
class Region(_Box):
    """A filled ground behind a group of nodes. NOT a `Node`.

    The distinction is the whole point: `clearance_violations` treats an
    edge entering an unattached `Node` as a collision, and a region is the
    one box an edge is SUPPOSED to run through. Keeping them separate types
    means neither check has to special-case the other.

    Members are named explicitly (`enclose` records them) so that
    "this group contains these nodes" is a claim the program makes and
    `region_containment_violations` can refute. A derived bbox that always
    fit would check nothing.
    """

    key: str
    center: XY
    width: float
    height: float
    members: tuple[str, ...] = ()
    label: str | None = None
    role: Role = Role.CONSTRUCTION     # background structure, per the grammar
    corner: float | None = None        # None -> 0.10*min(w,h): softer than a Node
    fill: str | None = None            # None -> the role's ink at `opacity`
    opacity: float = REGION_OPACITY
    stroked: bool = False              # a drawn border instead of a bare wash
    label_role: Role = Role.ANNOTATION
    label_size_pt: float | None = None
    label_ha: str = "center"
    label_va: str = "top"              # a group title, inside its own top edge
    label_offset_px: XY = (0.0, 12.0)  # canvas px, +y DOWN
    n_arc: int = 6

    @property
    def radius(self) -> float:
        r = self.corner if self.corner is not None else 0.10 * min(self.width, self.height)
        return float(min(max(r, 0.0), 0.5 * min(self.width, self.height)))

    @property
    def area(self) -> float:
        return float(abs(self.width) * abs(self.height))

    def items(self) -> list[Item]:
        out: list[Item] = [FilledCurve(self.outline(), role=self.role,
                                       color=self.fill, opacity=self.opacity,
                                       outline=self.stroked)]
        if self.label:
            x0, y0, x1, y1 = self.rect
            cx = {"left": x0, "center": 0.5 * (x0 + x1), "right": x1}[self.label_ha]
            out.append(MathLabel(self.label, (cx, y1 if self.label_va == "top" else y0),
                                 role=self.label_role, ha=self.label_ha,
                                 va=self.label_va, size_pt=self.label_size_pt,
                                 offset_px=self.label_offset_px))
        return out


def bbox(objs: Sequence, pad: float = 0.0) -> tuple[float, float, float, float]:
    """(x0, y0, x1, y1) enclosing Nodes, Regions and raw points, plus pad."""
    xs: list[float] = []
    ys: list[float] = []
    for o in objs:
        if isinstance(o, _Box):
            x0, y0, x1, y1 = o.rect
            xs += [x0, x1]
            ys += [y0, y1]
        else:
            p = _as_xy(o)
            xs += [float(p[0])]
            ys += [float(p[1])]
    if not xs:
        raise ValueError("bbox of nothing")
    return (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)


def enclose(key: str, members: Sequence, pad: float = 0.0, **kw) -> Region:
    """A `Region` sized to its members' bbox, recording their keys.

    Members may be Nodes, Regions or bare points; only the ones with a key
    are recorded, because only those can be checked afterwards. `pad` is
    the clearance in math units — the visible margin of the group.
    """
    x0, y0, x1, y1 = bbox(members, pad)
    return Region(key, (0.5 * (x0 + x1), 0.5 * (y0 + y1)), x1 - x0, y1 - y0,
                  members=tuple(m.key for m in members if isinstance(m, _Box)),
                  **kw)


# --- typed edges ------------------------------------------------------------


@dataclass(frozen=True)
class EdgeDecor:
    """The decoration policy for one edge kind, expressed only in Curve
    fields the renderer already owns."""

    dash: str | None = "solid"        # "solid" forces solid over a role's dash
    head: str = "filled"              # filled | hollow | bar | none
    chevrons: int = 1                 # stacked heads at the tip
    width_scale: float = 1.0
    head_scale: float = 1.0


# The vocabulary. Five kinds, each earning its slot by a distinction the
# corpus actually makes:
#   excite / inhibit  the signed pair of transformer-circuits' evidence
#                     panels (pointed arrow vs flat-capped bar, borrowed
#                     from gene-regulation notation);
#   attend            a READ — where a component looks. Nothing is written,
#                     so it is dashed and thinner, with a hollow head;
#   copy              content carried through UNCHANGED — a double chevron,
#                     the same thing arriving twice;
#   map               content TRANSFORMED on the way (hollow single head).
# Adding a sixth kind should require a mechanism none of these expresses.
EDGE_KINDS: dict[str, EdgeDecor] = {
    "excite":  EdgeDecor(head="filled"),
    "inhibit": EdgeDecor(head="bar", chevrons=0),
    "map":     EdgeDecor(head="hollow"),
    "copy":    EdgeDecor(head="filled", chevrons=2),
    "attend":  EdgeDecor(dash="dashed", head="hollow", width_scale=0.85),
}


@dataclass
class Edge:
    """A routed polyline with a kind. `anchors` are the port points the
    caller asked for; `pts` is what gets drawn (identical unless trimmed)."""

    pts: np.ndarray
    kind: str
    anchors: tuple[XY, XY]
    key: str = ""
    role: Role = Role.CONTENT
    color: str | None = None
    opacity: float = 1.0
    width_scale: float = 1.0
    dash: str | None = None            # override the kind's dash (identity channel)
    head_scale: float = 1.0
    casing: bool = False
    # "this continues and I am choosing to stop": the terminal head is
    # replaced by a hollow diamond and an ellipsis past the tip, so the cut
    # is drawn rather than confessed in the caption.
    truncated: bool = False
    bar_half: float = BAR_HALF
    chevron_gap: float = CHEVRON_GAP
    trunc_half: float = TRUNC_DIAMOND_HALF
    label: str | None = None
    label_at: float = 0.5              # arc-length fraction, unless...
    label_anchor: XY | None = None     # ...an explicit point is given
    label_role: Role | None = None
    label_size_pt: float | None = None
    label_ha: str = "center"
    label_va: str = "bottom"
    label_offset_px: XY = (0.0, 0.0)
    label_halo: bool = False

    @property
    def decor(self) -> EdgeDecor:
        try:
            return EDGE_KINDS[self.kind]
        except KeyError:
            raise ValueError(
                f"unknown edge kind {self.kind!r}; have {sorted(EDGE_KINDS)}") from None

    @property
    def length(self) -> float:
        return float(_arclen(self.pts)[-1])

    def point_at(self, t: float) -> XY:
        return point_at(self.pts, t)

    def head_fractions(self) -> tuple[float, ...]:
        """Arc-length fractions of the terminal heads. Stacked chevrons are
        a fixed MATH distance apart, not a fixed fraction — otherwise a long
        edge's second head drifts to mid-curve and reads as a waypoint."""
        d = self.decor
        if self.truncated:
            return ()                  # the diamond IS the terminal mark
        if d.head not in ("filled", "hollow") or d.chevrons <= 0:
            return ()
        L = self.length or 1.0
        fr = [max(0.0, 1.0 - i * self.chevron_gap / L) for i in range(d.chevrons)]
        return tuple(sorted(fr))

    def _terminal_tangent(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(tip, unit tangent, unit normal) at the far end of the route."""
        d = self.pts[-1] - self.pts[-2]
        u = d / (float(np.hypot(*d)) or 1.0)
        return self.pts[-1], u, np.array([-u[1], u[0]])

    def _terminal_bar(self) -> Curve:
        tip, _, nrm = self._terminal_tangent()
        return Curve(np.vstack([tip + self.bar_half * nrm, tip - self.bar_half * nrm]),
                     role=self.role, color=self.color, opacity=self.opacity,
                     width_scale=self.width_scale * self.decor.width_scale,
                     dash="solid")

    def _truncation_mark(self) -> list[Item]:
        """Hollow diamond on the tip, `\\cdots` just past it.

        Literal points, not derived ink: the diamond is stated in math
        units like the `inhibit` bar, so it scales with the schematic and
        no canvas-px resolver in render.py has to know it exists.
        """
        tip, u, nrm = self._terminal_tangent()
        h = self.trunc_half
        dia = np.vstack([tip + h * u, tip + h * nrm, tip - h * u, tip - h * nrm])
        lab = tip + 2.5 * h * u
        return [Curve(dia, role=self.role, color=self.color,
                      opacity=self.opacity, closed=True, dash="solid",
                      width_scale=self.width_scale * self.decor.width_scale),
                MathLabel(r"\cdots", (float(lab[0]), float(lab[1])),
                          role=Role.ANNOTATION, ha="center", va="center")]

    def items(self) -> list[Item]:
        d = self.decor
        out: list[Item] = [Curve(
            self.pts, role=self.role, color=self.color, opacity=self.opacity,
            width_scale=self.width_scale * d.width_scale,
            dash=self.dash if self.dash is not None else d.dash,
            arrows=self.head_fractions(),
            arrow_style="hollow" if d.head == "hollow" else "filled",
            arrow_scale=self.head_scale * d.head_scale,
            casing=self.casing)]
        if self.truncated:
            out += self._truncation_mark()
        elif d.head == "bar":
            out.append(self._terminal_bar())
        if self.label:
            anchor = self.label_anchor or self.point_at(self.label_at)
            out.append(MathLabel(self.label, anchor,
                                 role=self.label_role or self.role,
                                 ha=self.label_ha, va=self.label_va,
                                 size_pt=self.label_size_pt,
                                 offset_px=self.label_offset_px,
                                 halo=self.label_halo))
        return out


# --- routing (chosen, never solved) -----------------------------------------


def straight(p, q) -> np.ndarray:
    return np.vstack([_as_xy(p), _as_xy(q)])


def elbow(p, q, corner="hv") -> np.ndarray:
    """One axis-aligned bend. `corner` is either the bend point itself or
    `"hv"` (horizontal first) / `"vh"` (vertical first)."""
    a, b = _as_xy(p), _as_xy(q)
    if isinstance(corner, str):
        if corner == "hv":
            c = np.array([b[0], a[1]])
        elif corner == "vh":
            c = np.array([a[0], b[1]])
        else:
            raise ValueError(f"elbow corner must be 'hv', 'vh', or a point; got {corner!r}")
    else:
        c = _as_xy(corner)
    return np.vstack([a, c, b])


def quad_through(p, q, via, n: int = 64) -> np.ndarray:
    """Quadratic Bezier from p to q that PASSES THROUGH `via` at t = 1/2.

    The via point is where the curve goes, not an abstract control point —
    a figure program says "route this over y = 3.9", and it does."""
    a, b, m = _as_xy(p), _as_xy(q), _as_xy(via)
    c = 2.0 * m - 0.5 * (a + b)
    t = np.linspace(0.0, 1.0, max(int(n), 3))[:, None]
    return (1 - t) ** 2 * a + 2 * t * (1 - t) * c + t ** 2 * b


def cubic_through(p, q, via1, via2, n: int = 96) -> np.ndarray:
    """Cubic Bezier through `via1` at t = 1/3 and `via2` at t = 2/3 — the
    S-curve, with both waypoints stated rather than guessed."""
    a, b = _as_xy(p), _as_xy(q)
    m1, m2 = _as_xy(via1), _as_xy(via2)
    # Bernstein weights at t = 1/3 and 2/3, solved for the control points.
    A = np.array([[12.0, 6.0], [6.0, 12.0]]) / 27.0
    rhs = np.vstack([m1 - (8.0 * a + b) / 27.0, m2 - (a + 8.0 * b) / 27.0])
    c1, c2 = np.linalg.solve(A, rhs)
    t = np.linspace(0.0, 1.0, max(int(n), 4))[:, None]
    return ((1 - t) ** 3 * a + 3 * (1 - t) ** 2 * t * c1
            + 3 * (1 - t) * t ** 2 * c2 + t ** 3 * b)


def route_pts(p, q, route: str = "straight", *, via=None, corner="hv",
              n: int = 64) -> np.ndarray:
    if route == "straight":
        return straight(p, q)
    if route == "elbow":
        return elbow(p, q, corner)
    if route == "quad":
        if via is None:
            raise ValueError("route='quad' needs an explicit via point")
        return quad_through(p, q, via, n)
    if route == "cubic":
        if via is None or len(via) != 2:
            raise ValueError("route='cubic' needs via=(P1, P2)")
        return cubic_through(p, q, via[0], via[1], n)
    raise ValueError(f"unknown route {route!r}; have straight|elbow|quad|cubic")


def edge(p, q, kind: str, *, route: str = "straight", via=None, corner="hv",
         n: int = 64, **kw) -> Edge:
    """A typed edge from port point p to port point q.

    Routing is explicit: straight, one elbow at `corner`, or a Bezier
    through `via`. Everything else (role, colour, label, dash override)
    passes through to `Edge`.
    """
    if kind not in EDGE_KINDS:
        raise ValueError(f"unknown edge kind {kind!r}; have {sorted(EDGE_KINDS)}")
    pts = route_pts(p, q, route, via=via, corner=corner, n=n)
    return Edge(pts=pts, kind=kind,
                anchors=((float(pts[0][0]), float(pts[0][1])),
                         (float(pts[-1][0]), float(pts[-1][1]))), **kw)


# --- rails and grid placement -----------------------------------------------


def rails(positions: Sequence[float], extent: tuple[float, float], *,
          axis: str = "vertical", role: Role = Role.CONSTRUCTION,
          dash: str | None = None, width_scale: float = 1.0,
          opacity: float = 1.0,
          labels: Sequence[str] | None = None, label_end: str = "low",
          label_role: Role = Role.ANNOTATION, label_size_pt: float | None = None,
          label_offset_px: XY = (0.0, 10.0)) -> list[Item]:
    """The transformer-circuits idiom: one background rail per position.

    Rails are `Role.CONSTRUCTION` — grammar.md's ink hierarchy puts
    background structure there (dashed), reserving FRAME for page
    furniture. `labels` prints a header per rail at the `low` (default) or
    `high` end of the extent; `label_offset_px` is canvas px, +y DOWN.
    """
    if axis not in ("vertical", "horizontal"):
        raise ValueError(f"rail axis must be vertical|horizontal, got {axis!r}")
    lo, hi = float(extent[0]), float(extent[1])
    out: list[Item] = []
    for k, c in enumerate(positions):
        c = float(c)
        pts = (np.array([[c, lo], [c, hi]]) if axis == "vertical"
               else np.array([[lo, c], [hi, c]]))
        out.append(Curve(pts, role=role, dash=dash, width_scale=width_scale,
                         opacity=opacity))
    if labels is not None:
        end = lo if label_end == "low" else hi
        va = "top" if (axis == "vertical") == (label_end == "low") else "bottom"
        for c, latex in zip(positions, labels):
            anchor = ((float(c), end) if axis == "vertical" else (end, float(c)))
            out.append(MathLabel(latex, anchor, role=label_role,
                                 ha="center" if axis == "vertical" else
                                 ("right" if label_end == "low" else "left"),
                                 va=va if axis == "vertical" else "center",
                                 size_pt=label_size_pt,
                                 offset_px=label_offset_px))
    return out


def columns(n: int, x0: float = 0.0, dx: float = 1.0) -> list[float]:
    """Column centers x0, x0+dx, ... — dumb affine, no packing."""
    return [x0 + i * dx for i in range(int(n))]


def rows(n: int, y0: float = 0.0, dy: float = 1.0) -> list[float]:
    """Row centers y0, y0+dy, ... (+dy goes UP, math coords)."""
    return [y0 + j * dy for j in range(int(n))]


@dataclass(frozen=True)
class Grid:
    """(i, j) -> math-coord center. An affine map and nothing else."""

    origin: XY = (0.0, 0.0)
    dx: float = 1.0
    dy: float = 1.0

    def x(self, i: float) -> float:
        return self.origin[0] + i * self.dx

    def y(self, j: float) -> float:
        return self.origin[1] + j * self.dy

    def center(self, i: float, j: float) -> XY:
        return (self.x(i), self.y(j))


def items(*objs) -> list[Item]:
    """Flatten Nodes / Edges / nested sequences / raw Items into a scene
    list: `scene.add(*schematic.items(nodes, edges, rail_items))`."""
    out: list[Item] = []
    for o in objs:
        if isinstance(o, (Node, Edge)):
            out.extend(o.items())
        elif isinstance(o, (list, tuple)):
            out.extend(items(*o))
        else:
            out.append(o)
    return out


# --- schematic checks (assertion helpers, not magic) ------------------------


def label_extent_px(latex: str, size_pt: float = 11.0) -> tuple[float, float]:
    """Exact typeset (width, height) in canvas px — the same metrics the
    mechanical gate boxes with."""
    m = render_math(latex, size_pt)
    return m.width_px, m.height_px


def px_per_unit(xlim: tuple[float, float], width_px: float,
                pad_frac: float = 0.08) -> float:
    """Math units -> canvas px, mirroring layout.Transform for a scene whose
    xlim is explicit. Needed to compare a typeset label against a box."""
    dx = max(float(xlim[1]) - float(xlim[0]), 1e-12) * (1.0 + 2.0 * pad_frac)
    return float(width_px) / dx


def label_overflow(nodes: Sequence[Node], scale: float, size_pt: float,
                   pad_px: float = 6.0) -> list[str]:
    """Nodes whose label does not fit inside its own box at `scale` px/unit.

    Sizes are declared, so this is the check that keeps the declaration
    honest — it does not resize anything.
    """
    bad: list[str] = []
    for nd in nodes:
        if not nd.label:
            continue
        w, h = label_extent_px(nd.label, nd.label_size_pt or size_pt)
        bw, bh = nd.width * scale, nd.height * scale
        if w + pad_px > bw or h + pad_px > bh:
            bad.append(f"{nd.key}: label {w:.0f}x{h:.0f}px in a "
                       f"{bw:.0f}x{bh:.0f}px box (pad {pad_px:.0f})")
    return bad


def region_containment_violations(regions: Sequence[Region],
                                  nodes: Sequence[Node],
                                  pad: float = 0.0) -> list[str]:
    """Declared members whose rect is not fully inside their region.

    A group that visibly clips one of its own members is a content bug in a
    Class B figure — the reader reads membership off the fill, so a box half
    outside says the wrong thing about the mechanism.
    """
    by_key = {nd.key: nd for nd in nodes}
    bad: list[str] = []
    for r in regions:
        rx0, ry0, rx1, ry1 = r.rect
        for k in r.members:
            nd = by_key.get(k)
            if nd is None:
                bad.append(f"region {r.key!r} claims member {k!r}, which is "
                           f"not among the nodes")
                continue
            x0, y0, x1, y1 = nd.rect
            if (x0 < rx0 - pad or y0 < ry0 - pad
                    or x1 > rx1 + pad or y1 > ry1 + pad):
                bad.append(f"member {k!r} escapes region {r.key!r}: node rect "
                           f"{tuple(round(v, 3) for v in nd.rect)} outside "
                           f"{tuple(round(v, 3) for v in r.rect)}")
    return bad


def region_nesting_violations(regions: Sequence[Region],
                              tol: float = 1e-9) -> list[str]:
    """Region pairs that half-overlap — neither nested nor disjoint.

    Containment is only readable if the regions form a TREE. Two grounds
    that share a corner make a third, unnamed region out of the overlap and
    the reader has no way to know what it means; there is no honest way to
    draw it, so it is a violation rather than a style note.
    """
    bad: list[str] = []
    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            a, b = regions[i], regions[j]
            ax0, ay0, ax1, ay1 = a.rect
            bx0, by0, bx1, by1 = b.rect
            if (ax1 <= bx0 + tol or bx1 <= ax0 + tol
                    or ay1 <= by0 + tol or by1 <= ay0 + tol):
                continue                                   # disjoint
            a_in_b = (bx0 - tol <= ax0 and ax1 <= bx1 + tol
                      and by0 - tol <= ay0 and ay1 <= by1 + tol)
            b_in_a = (ax0 - tol <= bx0 and bx1 <= ax1 + tol
                      and ay0 - tol <= by0 and by1 <= ay1 + tol)
            if not (a_in_b or b_in_a):
                bad.append(f"regions {a.key!r} and {b.key!r} half-overlap — "
                           f"grouping grounds must nest or be disjoint")
    return bad


def attached_nodes(e: Edge, nodes: Sequence[Node], tol: float) -> list[str]:
    """Keys of the nodes an edge's anchors land on (within tol)."""
    return [nd.key for nd in nodes
            if any(nd.boundary_distance(a) <= tol for a in e.anchors)]


def port_offsets(edges: Sequence[Edge], nodes: Sequence[Node]
                 ) -> list[tuple[str, int, float]]:
    """(edge key, anchor index, distance to the NEAREST node boundary) for
    every edge anchor. Port attachment is exact when these are ~0."""
    out: list[tuple[str, int, float]] = []
    for e in edges:
        for i, a in enumerate(e.anchors):
            d = min((nd.boundary_distance(a) for nd in nodes), default=float("inf"))
            out.append((e.key or e.kind, i, float(d)))
    return out


def max_port_offset(edges: Sequence[Edge], nodes: Sequence[Node]) -> float:
    offs = port_offsets(edges, nodes)
    return max((d for _, _, d in offs), default=0.0)


def _seg_clip_rect(p, q, rect: tuple[float, float, float, float], pad: float = 0.0):
    """The part of segment p->q inside the padded rect (Liang-Barsky), or
    None if it misses. Returns the two clipped endpoints."""
    x0, y0, x1, y1 = rect[0] - pad, rect[1] - pad, rect[2] + pad, rect[3] + pad
    d = np.asarray(q, dtype=float) - np.asarray(p, dtype=float)
    t0, t1 = 0.0, 1.0
    for den, num in ((-d[0], p[0] - x0), (d[0], x1 - p[0]),
                     (-d[1], p[1] - y0), (d[1], y1 - p[1])):
        if den == 0.0:
            if num < 0.0:
                return None
            continue
        r = num / den
        if den < 0.0:
            t0 = max(t0, r)
        else:
            t1 = min(t1, r)
        if t0 > t1:
            return None
    a = np.asarray(p, dtype=float) + t0 * d
    b = np.asarray(p, dtype=float) + t1 * d
    return a, b


def clearance_violations(edges: Sequence[Edge], nodes: Sequence[Node],
                         pad: float = 0.0, attach_tol: float = 1e-9
                         ) -> list[str]:
    """Edges whose polyline enters the padded rect of a node it is not
    attached to — the schematic version of a label collision.

    Attachment is decided by the edge's own anchors, so an edge that ends
    ON a box is allowed to touch it and only that one.

    Tested per SEGMENT, not per vertex: a straight edge is two points, and
    a vertex test would let it pass clean through a box between them.
    """
    bad: list[str] = []
    for e in edges:
        skip = set(attached_nodes(e, nodes, attach_tol))
        for nd in nodes:
            if nd.key in skip:
                continue
            for k in range(len(e.pts) - 1):
                seg = _seg_clip_rect(e.pts[k], e.pts[k + 1], nd.rect, pad)
                if seg is None:
                    continue
                x, y = 0.5 * (seg[0] + seg[1])
                bad.append(f"edge {e.key or e.kind!r} enters node {nd.key!r} "
                           f"near ({x:.2f}, {y:.2f}) (pad {pad:g})")
                break
    return bad


def _seg_cross(p1, p2, p3, p4, eps: float = 1e-9):
    """Proper crossing point of two segments, or None. Endpoint touches and
    collinear overlaps do not count — edges meeting at a shared port are
    joined, not crossed."""
    r = p2 - p1
    s = p4 - p3
    den = r[0] * s[1] - r[1] * s[0]
    if abs(den) < 1e-15:
        return None
    d = p3 - p1
    t = (d[0] * s[1] - d[1] * s[0]) / den
    u = (d[0] * r[1] - d[1] * r[0]) / den
    if eps < t < 1.0 - eps and eps < u < 1.0 - eps:
        return p1 + t * r
    return None


def crossings(edges: Sequence[Edge], merge_tol: float = 1e-6
              ) -> list[tuple[str, str, XY]]:
    """Every place two distinct edges cross, deduplicated per pair.

    Crossings are the honest cost of a routing choice; the figure program
    declares how many it accepts and asserts the count, so a later edit
    that introduces spaghetti fails the numerical gate.
    """
    out: list[tuple[str, str, XY]] = []
    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            a, b = edges[i], edges[j]
            hits: list[np.ndarray] = []
            for k in range(len(a.pts) - 1):
                for m in range(len(b.pts) - 1):
                    x = _seg_cross(a.pts[k], a.pts[k + 1], b.pts[m], b.pts[m + 1])
                    if x is None:
                        continue
                    if all(np.hypot(*(x - h)) > merge_tol for h in hits):
                        hits.append(x)
            for h in hits:
                out.append((a.key or a.kind, b.key or b.kind,
                            (float(h[0]), float(h[1]))))
    return out


def crossing_count(edges: Sequence[Edge], merge_tol: float = 1e-6) -> int:
    return len(crossings(edges, merge_tol))


# ============================================================================
# The layout pass
# ============================================================================
#
# Everything above states coordinates. This half computes three of them —
# where a box's boundary is, how big a box has to be to hold its own label,
# and where a layered graph's ranks fall — and nothing else. The distinction
# that matters is between DERIVED and SOLVED: a boundary intersection and a
# longest-path rank are functions of what the program already said, with one
# answer and no objective; a force-directed position or a routed detour is a
# search over layouts, which is exactly what stays out (primitive-gaps.md:
# model proposes, gates dispose). Every computed position can be replaced by
# an explicit one, and the override is local — it never triggers a re-solve.


# --- boundary attachment ----------------------------------------------------
#
# An edge drawn between two centers passes UNDER both boxes and, with a
# paper fill, reappears as a line growing out of the middle of a word. Edges
# terminate on the boundary instead, which is also what makes the arrowhead
# land where the reader looks for it: on the box, not in it.


@dataclass(frozen=True)
class Port:
    """A named boundary anchor bound to its node — `Port(mha, "top")`.

    Sugar over `Node.port`, worth its ten lines because it lets an endpoint
    be passed around as one object: `connect(Port(a, "right"), b, ...)`.
    """

    node: Node
    spec: object = "top"

    @property
    def xy(self) -> XY:
        return self.node.port(self.spec)


def _half_extents(node: Node, pad: float) -> tuple[float, float]:
    return (node.width / 2.0 + pad, node.height / 2.0 + pad)


def boundary_toward(node: Node, target, pad: float = 0.0) -> XY:
    """Where the ray from `node`'s center toward `target` leaves its rect.

    The rect, not the rounded outline — `Node.boundary_distance` measures
    against the rect for the same reason (rounding is cosmetic, and an
    anchor pulled inside the corner arc reads as detached from its box).
    """
    cx, cy = node.center
    tx, ty = float(target[0]), float(target[1])
    dx, dy = tx - cx, ty - cy
    if dx == 0.0 and dy == 0.0:
        return (cx, cy)
    hw, hh = _half_extents(node, pad)
    t = min(hw / abs(dx) if dx else float("inf"),
            hh / abs(dy) if dy else float("inf"))
    return (cx + t * dx, cy + t * dy)


def _inside(p, node: Node, pad: float, eps: float) -> bool:
    """Strictly inside the padded rect. A point ON the boundary is not, so
    an endpoint that is already a port survives trimming untouched."""
    hw, hh = _half_extents(node, pad)
    return (abs(float(p[0]) - node.center[0]) < hw - eps
            and abs(float(p[1]) - node.center[1]) < hh - eps)


def _rect_exit(a: np.ndarray, b: np.ndarray, node: Node, pad: float) -> np.ndarray:
    """Exit point of segment a->b through the padded rect, with a inside.

    Slab method: a inside means the inside-interval [t0, t1] contains 0, so
    the exit is t1 = the nearest forward slab bound.
    """
    hw, hh = _half_extents(node, pad)
    cx, cy = node.center
    d = b - a
    t = float("inf")
    for da, ca, half, aa in ((d[0], cx, hw, a[0]), (d[1], cy, hh, a[1])):
        if da == 0.0:
            continue
        t = min(t, ((ca + half if da > 0 else ca - half) - aa) / da)
    if not np.isfinite(t):
        return b
    return a + max(min(t, 1.0), 0.0) * d


def trim_at_boundary(pts: np.ndarray, node: Node, end: str = "start",
                     pad: float = 0.0, eps: float = 1e-12) -> np.ndarray:
    """Cut a routed polyline back to `node`'s boundary at one end.

    Vertices inside the box are dropped and replaced by the exact crossing.
    A polyline that never enters the box comes back unchanged, so calling
    this is always safe and never moves an endpoint that was already a port.
    """
    pts = np.asarray(pts, dtype=float)
    if end == "end":
        return trim_at_boundary(pts[::-1], node, "start", pad, eps)[::-1]
    if end != "start":
        raise ValueError(f"end must be 'start' or 'end', got {end!r}")
    if len(pts) < 2 or not _inside(pts[0], node, pad, eps):
        return pts
    j = next((k for k in range(1, len(pts))
              if not _inside(pts[k], node, pad, eps)), None)
    if j is None:                      # the whole route is inside the box
        return pts
    return np.vstack([_rect_exit(pts[j - 1], pts[j], node, pad), pts[j:]])


def _endpoint(obj) -> tuple[Node, object | None]:
    """Node | Port | (node, port_spec) -> (node, port spec or None)."""
    if isinstance(obj, Port):
        return obj.node, obj.spec
    if isinstance(obj, (tuple, list)) and len(obj) == 2 and isinstance(obj[0], Node):
        return obj[0], obj[1]
    if isinstance(obj, Node):
        return obj, None
    raise TypeError(f"expected a Node, Port, or (Node, port); got {obj!r}")


def connect(src, dst, kind: str, *, route: str = "straight", via=None,
            corner="hv", src_port=None, dst_port=None, pad: float = 0.0,
            n: int = 64, **kw) -> Edge:
    """A typed edge between two NODES, terminating on both boundaries.

    Without a port spec the attachment face is derived: the ray toward the
    other node's center. With one, the caller's port wins — which is how
    fan-in stays legible (`connect(a, b, "map", dst_port="left@0.25")`).
    Routing is the same explicit menu as `edge`; the route is then trimmed
    back to each boundary so a Bezier that bulges over a box still starts
    and ends exactly on it.
    """
    src_node, src_spec = _endpoint(src)
    dst_node, dst_spec = _endpoint(dst)
    src_spec = src_port if src_port is not None else src_spec
    dst_spec = dst_port if dst_port is not None else dst_spec
    p = (src_node.port(src_spec) if src_spec is not None
         else boundary_toward(src_node, dst_node.center, pad))
    q = (dst_node.port(dst_spec) if dst_spec is not None
         else boundary_toward(dst_node, src_node.center, pad))
    pts = route_pts(p, q, route, via=via, corner=corner, n=n)
    pts = trim_at_boundary(pts, src_node, "start", pad)
    pts = trim_at_boundary(pts, dst_node, "end", pad)
    if kind not in EDGE_KINDS:
        raise ValueError(f"unknown edge kind {kind!r}; have {sorted(EDGE_KINDS)}")
    return Edge(pts=pts, kind=kind,
                anchors=((float(pts[0][0]), float(pts[0][1])),
                         (float(pts[-1][0]), float(pts[-1][1]))), **kw)


# --- layered ranking --------------------------------------------------------


def longest_path_ranks(keys: Sequence[str],
                       edges: Sequence[tuple[str, str]]) -> dict[str, int]:
    """rank(v) = length of the LONGEST path into v. Refuses cycles.

    Longest and not shortest, because a skip connection is the whole point
    of the graphs this draws: with shortest-path ranks a residual edge
    landing three ranks downstream would place its target beside its source
    and the skip would have to run backwards.
    """
    keys = list(keys)
    known = set(keys)
    succ: dict[str, list[str]] = {k: [] for k in keys}
    indeg: dict[str, int] = {k: 0 for k in keys}
    for u, v in edges:
        if u not in known or v not in known:
            raise ValueError(f"edge ({u!r}, {v!r}) names a node outside {keys}")
        succ[u].append(v)
        indeg[v] += 1
    rank = {k: 0 for k in keys}
    queue = [k for k in keys if indeg[k] == 0]
    seen = 0
    while queue:
        u = queue.pop(0)
        seen += 1
        for v in succ[u]:
            rank[v] = max(rank[v], rank[u] + 1)
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
    if seen != len(keys):
        stuck = sorted(k for k in keys if indeg[k] > 0)
        raise ValueError(f"cycle in the schematic graph, unranked: {stuck}")
    return rank


@dataclass(frozen=True)
class RankLayout:
    """The affine part: (rank, lane) -> math coords, and nothing else.

    `axis` is the FLOW direction. Vertical flow puts rank on +y (rank 0 at
    the bottom, math coords) and lane on +x; horizontal flow swaps them.
    Reverse a flow by making the gap negative — there is no direction flag,
    because a sign already is one.
    """

    axis: str = "vertical"
    rank_gap: float = 1.0
    lane_gap: float = 1.0
    origin: XY = (0.0, 0.0)

    def center(self, rank: float, lane: float) -> XY:
        r, l = float(rank) * self.rank_gap, float(lane) * self.lane_gap
        ox, oy = self.origin
        if self.axis == "vertical":
            return (ox + l, oy + r)
        if self.axis == "horizontal":
            return (ox + r, oy + l)
        raise ValueError(f"axis must be vertical|horizontal, got {self.axis!r}")


def rank_positions(keys: Sequence[str], *, layout: RankLayout | None = None,
                   ranks: dict[str, int] | None = None,
                   dag: Sequence[tuple[str, str]] = (),
                   lanes: dict[str, float] | None = None,
                   overrides: dict[str, XY] | None = None) -> dict[str, XY]:
    """Layered placement: key -> math-coord center.

    Three inputs, each strictly overriding the one before it, so a program
    can start with `dag=` and end with every position stated:

    1. `ranks` (or a longest-path ranking of `dag`) sets the flow coordinate;
    2. within a rank, keys carrying an explicit `lanes` entry keep it and
       the rest are spread evenly, centered on lane 0, in the order they
       appear in `keys`;
    3. `overrides` replaces a center outright.

    That is the whole algorithm. It packs nothing, avoids nothing, and
    minimizes no crossings — `crossings()` and `clearance_violations()`
    report what it produced and the figure program decides what to do.
    """
    layout = layout or RankLayout()
    lanes = dict(lanes or {})
    overrides = dict(overrides or {})
    keys = list(keys)
    if ranks is None:
        ranks = longest_path_ranks(keys, dag)
    missing = [k for k in keys if k not in ranks]
    if missing:
        raise ValueError(f"no rank for {missing}")

    by_rank: dict[int, list[str]] = {}
    for k in keys:
        by_rank.setdefault(int(ranks[k]), []).append(k)

    lane_of: dict[str, float] = {}
    for r, group in by_rank.items():
        free = [k for k in group if k not in lanes]
        for i, k in enumerate(free):
            lane_of[k] = i - (len(free) - 1) / 2.0
        for k in group:
            if k in lanes:
                lane_of[k] = float(lanes[k])

    return {k: (overrides[k] if k in overrides
                else layout.center(ranks[k], lane_of[k])) for k in keys}


# --- boxes sized from their own text ----------------------------------------
#
# `Node` still takes a stated size, and `label_overflow` still checks it.
# What auto-sizing removes is the ritual of tuning that number by hand at a
# scale the program already knows: `scale` comes from `px_per_unit(xlim,
# width)`, so this is a unit conversion of a measured glyph box, not a
# solver. It is a function returning a number the caller passes in — the
# caller can print it, round it, or ignore it.


# Clearance around the typeset label, in canvas px (total, both sides).
# Comfortably above `label_overflow`'s 6 px honesty margin.
AUTO_PAD_PX: XY = (26.0, 22.0)


def auto_size(latex: str, scale: float, size_pt: float = 11.0,
              pad_px: XY = AUTO_PAD_PX,
              min_size: XY = (0.0, 0.0)) -> tuple[float, float]:
    """(width, height) in MATH units for a box holding `latex` with padding.

    `scale` is px per math unit (`px_per_unit`). `min_size` is what makes a
    column of boxes uniform: size each label, take the max, pass it back in.
    """
    w_px, h_px = label_extent_px(latex, size_pt)
    return (max((w_px + float(pad_px[0])) / scale, float(min_size[0])),
            max((h_px + float(pad_px[1])) / scale, float(min_size[1])))


def auto_node(key: str, center: XY, label: str, *, scale: float,
              size_pt: float = 11.0, pad_px: XY = AUTO_PAD_PX,
              min_size: XY = (0.0, 0.0), **kw) -> Node:
    """`Node` whose size is `auto_size(label, ...)`; everything else passes
    through. `label_size_pt` in **kw is honored for the measurement too."""
    pt = kw.get("label_size_pt") or size_pt
    w, h = auto_size(label, scale, pt, pad_px, min_size)
    return Node(key, center, w, h, label=label, **kw)


def circle_node(key: str, center: XY, radius: float, label: str | None = None,
                *, n_arc: int = 12, **kw) -> Node:
    """A round node — the ⊕ of a residual add, a state of a Markov chain.

    A `Node` with width = height = 2r and the corner radius maxed, so ports,
    boundary distance, and clipping all keep working: they measure the rect,
    which for a circle is its circumscribed square. An anchor on a diagonal
    therefore sits a hair outside the drawn circle; at these radii that is
    below a stroke width, and the alternative is a second shape model.
    """
    d = 2.0 * float(radius)
    return Node(key, center, d, d, label=label, corner=float(radius),
                n_arc=n_arc, **kw)


# --- assembly ---------------------------------------------------------------


def assemble(nodes: Sequence[Node], edges: Sequence[Edge], *,
             regions: Sequence[Region] = (),
             under: Sequence = (), over: Sequence = ()) -> list[Item]:
    """nodes + edges -> scene items, in the one draw order that reads.

    Edge strokes go down first and node boxes paint over them, so a
    paper-filled box masks whatever runs behind it and every edge appears to
    stop cleanly at the boundary even where it does not. Edge LABELS are
    lifted back above the boxes — an edge label is annotation, and being
    masked by a box downstream of it is the one failure this ordering would
    otherwise introduce. `under` (rails, background) and `over` (titles)
    bracket the whole thing.

    `regions` paint first, largest area first, so a nested region lands on
    top of its parent and the contrast ladder comes out right whatever order
    the program listed them in. Draw order is DERIVED from the geometry
    here, unlike everywhere else in this module, because the alternative is
    a hand-maintained depth integer that silently disagrees with the rects.
    """
    out: list[Item] = list(items(under))
    for r in sorted(regions, key=lambda r: -r.area):
        out.extend(r.items())
    edge_items = [it for e in edges for it in e.items()]
    out += [it for it in edge_items if not isinstance(it, MathLabel)]
    out += items(nodes)
    out += [it for it in edge_items if isinstance(it, MathLabel)]
    out += items(over)
    return out
