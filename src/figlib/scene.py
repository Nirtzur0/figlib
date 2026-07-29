"""Scene graph: typed primitives in math coordinates.

Primitives carry semantic roles, never raw stroke settings; only
layout.Transform ever converts to canvas pixels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .style import Role

XY = tuple[float, float]


@dataclass
class Curve:
    pts: np.ndarray                # (N, 2)
    role: Role = Role.CONTENT
    closed: bool = False
    width_scale: float = 1.0
    opacity: float = 1.0
    color: str | None = None       # semantic override (e.g. hue encodes a coordinate)
    # dash independent of role: "dashed"/"dotted" (style-resolved) or a raw
    # SVG dasharray — dash pattern can carry identity across panels
    dash: str | None = None
    # tangent-oriented direction markers at arc-length fractions of the
    # curve (Needham: filled head = contour/motion, hollow = streamline)
    arrows: tuple[float, ...] = ()
    arrow_style: str = "filled"    # filled | hollow
    arrow_scale: float = 1.0
    # width multipliers sampled at equal arc-length fractions, linearly
    # interpolated along the curve (x ink.width x width_scale). Renders as
    # a filled offset polygon; dash is ignored when set.
    width_profile: tuple[float, ...] | None = None
    # stroke-linecap override; None -> the default round. Ramp segments use
    # "butt" so translucent joints don't double-draw.
    cap: str | None = None
    # paper-colored under-stroke: this curve reads as passing OVER whatever
    # was drawn before it (cartographic casing). Skipped on transparent
    # themes; ignored with width_profile.
    casing: bool = False
    # correspondence name: the marks that ARE the same object across the
    # parts of a composite figure share a key. Renders nothing; it is what
    # correspond.py measures the residual against.
    key: str | None = None


@dataclass(frozen=True)
class Gradient:
    """A fill paint: linear or radial color ramp, axis in MATH coords.

    p0 -> p1 is the linear axis (radial: center -> a point on the rim).
    Coordinates are scene coords like every primitive; only
    layout.Transform converts them at emission.
    """
    stops: tuple[tuple[float, str], ...]   # (offset in [0,1], "#rrggbb")
    kind: str = "linear"                   # "linear" | "radial"
    p0: XY = (0.0, 0.0)
    p1: XY = (1.0, 0.0)

    @staticmethod
    def from_ramp(ramp: Callable[[float], str], p0: XY, p1: XY, *,
                  t_range: tuple[float, float] = (0.0, 1.0), n: int = 5,
                  kind: str = "linear") -> "Gradient":
        """Sample `ramp` over t_range into n evenly spaced stops."""
        t0, t1 = t_range
        stops = tuple((k / (n - 1), ramp(t0 + (t1 - t0) * k / (n - 1)))
                      for k in range(n))
        return Gradient(stops=stops, kind=kind,
                        p0=(float(p0[0]), float(p0[1])),
                        p1=(float(p1[0]), float(p1[1])))


@dataclass
class FilledCurve:
    pts: np.ndarray                # (N, 2), closed implicitly
    role: Role = Role.CONTENT
    opacity: float = 0.18
    outline: bool = True
    color: str | None = None       # fill override
    edge_color: str | None = None  # stroke override (used by 3D shading)
    edge_width: float | None = None
    # texture instead of tint: "stipple" | "hatch" | "crosshatch". Tile ink
    # in the resolved fill color on a TRANSPARENT ground (paper shows
    # through), so monochrome themes keep regions distinguishable; opacity
    # does not apply — texture is ink, not a wash.
    pattern: str | None = None
    # interior boundaries, each (N, 2) closed implicitly: one path with
    # multiple subpaths and fill-rule evenodd; outline strokes all of them
    holes: tuple[np.ndarray, ...] = ()
    # gradient fill paint (overrides color); axis in math coords
    gradient: Gradient | None = None
    # grain INSIDE the fill: 0 = none, else opacity of the grain tile
    # overlaid on this shape (skipped on transparent themes)
    grain: float = 0.0
    # correspondence name: the marks that ARE the same object across the
    # parts of a composite figure share a key. Renders nothing; it is what
    # correspond.py measures the residual against.
    key: str | None = None

    def __post_init__(self) -> None:
        if self.gradient is not None and self.pattern is not None:
            raise ValueError(
                "FilledCurve: gradient and pattern are competing fill paints")


@dataclass
class Vector:
    tail: XY
    tip: XY
    role: Role = Role.CONTENT
    width_scale: float = 1.0
    # False -> ghost copy: paper-filled head and shaft outline (the
    # white-original / black-image convention of amplitwist figures)
    filled: bool = True
    # correspondence name: the marks that ARE the same object across the
    # parts of a composite figure share a key. Renders nothing; it is what
    # correspond.py measures the residual against.
    key: str | None = None


@dataclass
class Point:
    xy: XY
    role: Role = Role.CONTENT
    filled: bool = True
    radius_scale: float = 1.0
    color: str | None = None       # fill/stroke override, same convention as Curve.color
    # correspondence name: the marks that ARE the same object across the
    # parts of a composite figure share a key. Renders nothing; it is what
    # correspond.py measures the residual against.
    key: str | None = None


@dataclass
class MathLabel:
    latex: str
    anchor: XY                     # math coords
    role: Role = Role.CONTENT
    color: str | None = None
    size_pt: float | None = None   # None -> style.label_size_pt
    ha: str = "left"               # left | center | right
    va: str = "base"               # top | center | base | bottom
    # nudge in canvas px. CAUTION: +y moves DOWN (canvas frame), the
    # opposite of the math-coords y the anchor lives in.
    offset_px: XY = (0.0, 0.0)
    # rotation about the (offset) anchor, degrees CCW on the page —
    # covers text-along-a-curve; the gate boxes the rotated rect's AABB
    angle_deg: float = 0.0
    # paper-colored casing behind the glyphs so the label stays legible on
    # busy ink (cartographic halo). Skipped on transparent themes.
    halo: bool = False
    # typographic REGISTER — a semantic channel, not appearance:
    # "mono" = literal model input / data, "sans" = human interpretation,
    # None = mathematics. See typeset.apply_register.
    register: str | None = None
    # this label's position IS meaning (e.g. text along a curve): the
    # auto-place pass never moves it, collisions fall to the gate
    pin: bool = False
    # correspondence name: the marks that ARE the same object across the
    # parts of a composite figure share a key. Renders nothing; it is what
    # correspond.py measures the residual against.
    key: str | None = None


@dataclass
class RightAngleMark:
    corner: XY
    dir1: XY                       # the two edge directions (math coords)
    dir2: XY
    size: float                    # side length in math units
    role: Role = Role.ANNOTATION
    # correspondence name: the marks that ARE the same object across the
    # parts of a composite figure share a key. Renders nothing; it is what
    # correspond.py measures the residual against.
    key: str | None = None


@dataclass
class AngleMark:
    center: XY
    dir1: XY
    dir2: XY
    radius: float                  # math units
    role: Role = Role.ANNOTATION
    # correspondence name: the marks that ARE the same object across the
    # parts of a composite figure share a key. Renders nothing; it is what
    # correspond.py measures the residual against.
    key: str | None = None


@dataclass
class Brace:
    """Curly brace spanning p1 -> p2 (a measured length), bulging to the
    left of the p1->p2 direction when side > 0, right when side < 0; the
    label is typeset just past the center cusp."""
    p1: XY
    p2: XY
    side: float = 1.0
    depth: float | None = None     # math units; None -> 6% of the span
    label: str | None = None
    role: Role = Role.ANNOTATION
    # correspondence name: the marks that ARE the same object across the
    # parts of a composite figure share a key. Renders nothing; it is what
    # correspond.py measures the residual against.
    key: str | None = None

    def frame(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """(midpoint, unit tangent, unit normal toward the bulge, depth) —
        the one place the brace's geometry is defined; layout and render
        both derive from it."""
        p1 = np.asarray(self.p1, dtype=float)
        p2 = np.asarray(self.p2, dtype=float)
        v = p2 - p1
        span = float(np.hypot(*v)) or 1.0
        u = v / span
        n = np.array([-u[1], u[0]]) * (1.0 if self.side >= 0 else -1.0)
        d = self.depth if self.depth is not None else 0.06 * span
        return (p1 + p2) / 2.0, u, n, d


@dataclass
class Callout:
    """Boxed label with a leader: rounded paper-filled rect centered at
    anchor, straight leader from the box edge nearest target, small filled
    arrowhead at target."""
    latex: str
    anchor: XY
    target: XY
    role: Role = Role.ANNOTATION
    boxed: bool = True
    size_pt: float | None = None
    # correspondence name: the marks that ARE the same object across the
    # parts of a composite figure share a key. Renders nothing; it is what
    # correspond.py measures the residual against.
    key: str | None = None


@dataclass
class RasterField:
    """Dense field embedded as an image — the honest encoding when there is
    no vector substitute at the density (attention maps, loss landscapes,
    PDE fields). Row 0 renders at the TOP of extent (y = y1).

    Two modes, told apart by ndim, because some fields have no scalar to
    normalize:

    - **(H, W) scalar** — normalized over [vmin, vmax] and mapped through
      ramp (default: the theme's ramp, else gray).
    - **(H, W, 3) sRGB in [0, 1]** — already color, drawn as given (clipped
      to the cube); vmin/vmax/ramp do not apply. This is the mode a *cyclic*
      channel needs: phase is periodic, so no monotone ramp can carry it and
      the producer resolves color per pixel through `theme.phase`. Keeping
      it on the same field means one item type, one renderer, one extent
      rule — a new channel, not a new primitive.
    """
    values: np.ndarray
    extent: tuple[float, float, float, float]   # (x0, x1, y0, y1)
    ramp: Callable[[float], str] | None = None
    vmin: float | None = None
    vmax: float | None = None
    opacity: float = 1.0
    interp: bool = False           # False -> image-rendering: pixelated
    # correspondence name: the marks that ARE the same object across the
    # parts of a composite figure share a key. Renders nothing; it is what
    # correspond.py measures the residual against.
    key: str | None = None


Item = (Curve | FilledCurve | Vector | Point | MathLabel | RightAngleMark
        | AngleMark | Brace | Callout | RasterField)


@dataclass
class Scene:
    items: list[Item] = field(default_factory=list)
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    # None -> equal aspect (geometry in the plane); a float -> canvas
    # height in px, axes scaled independently (e.g. (t, x) plots)
    height_px: float | None = None
    # clip every item: "frame" -> the xlim/ylim rect; clip_pts -> a closed
    # polygon in math coords (disc portraits). Rendered as one SVG
    # <clipPath> around the scene's item group; extents honor xlim/ylim
    # exactly as without clipping.
    clip: str | None = None
    clip_pts: np.ndarray | None = None

    def add(self, *items: Item) -> None:
        self.items.extend(items)
