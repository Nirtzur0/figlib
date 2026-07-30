"""Word-scale inset: embed one Scene's items into a host scene's coords.

`embed(small, at=..., width=...)` affinely maps the small scene's items
into HOST math coordinates and returns plain items for `host.add(*...)`.
It is a PRODUCER, not a renderer: the result goes through the one
rendering path, so themes and every gate apply to inset content for free.

The mapping IS the semantics:

- **Source rect required.** The small scene must have explicit
  `xlim`/`ylim` (the mapping needs a source rect) and must NOT set
  `height_px` — a px-height scene has no math aspect, so it has no
  honest place in another scene's math plane; both raise `ValueError`.
- **Equal aspect, one linear factor.** Dest rect is centered at `at`,
  `width` wide, `width * dy/dx` tall; a single scale `s = width/dx`
  applies to both axes.
- **Geometry scales, type and ink never do.** Coordinates and math-unit
  lengths (`Brace.depth`, `AngleMark.radius`, `RightAngleMark.size`)
  multiply by `s`. Canvas-px quantities — `size_pt`, `offset_px`,
  `width_scale`, `radius_scale`, `arrow_scale`, dash patterns,
  `width_profile`, pattern/grain tiles — pass through untouched: type
  and ink stay at reading size by the canvas-px invariant. A too-dense
  inset is caught by the mechanical gate; the fix is a bigger inset or
  less ink, never smaller type. Direction fields (`dir1`/`dir2`) and
  arc-length fractions (`Curve.arrows`) are invariant under a uniform
  scale and pass through unchanged.
- **Gradient axes are mapped.** `Gradient.p0/p1` are math coords that
  render resolves through the host Transform (`userSpaceOnUse`), so an
  unmapped axis would paint the fill from the WRONG place in the host.
- **Copies, never mutation.** Every produced item is a
  `dataclasses.replace` copy with freshly computed arrays; the small
  scene is left intact and reusable.
- **Unknown item types fail loudly.** The dispatch table covers every
  type in `scene.Item`; a future primitive arriving here unmapped
  raises `TypeError` naming it instead of passing through with stale
  coordinates.

Furniture (kept minimal — the thumbnail is the content):

- `frame=True` emits ONE closed `Role.FRAME` Curve on the dest rect,
  corners rounded at 6% of the shorter dest side (a proportion, not a
  math-unit constant, so thumbnails stay self-similar at any width; the
  FRAME role's hairline-dash ink keeps it quieter than any content).
- `leader_to` emits ONE `Role.ANNOTATION` Vector from the nearest point
  of the dest rect boundary to the target. A Vector, not literal
  arrowhead points: `render._arrowhead` is canvas-px glyph math —
  figure.py can call it lazily because connector coordinates ARE canvas
  px, but here coordinates are host MATH coords, and a head triangle
  computed in math units would scale with the host transform, breaking
  the type/ink invariant. Vector already resolves its head in canvas px
  at render time through the single rendering path, so the head stays at
  reading size whatever the inset or host scale.
- `key` (when given) is stamped on every produced item — the inset
  participates in correspondence as ONE object. When omitted, the small
  scene's own per-item keys survive the mapping.

Known limit (v1): inset content is NOT clipped to its frame. Content
computed past the small scene's xlim/ylim leaks into the host; the frame
is FRAME-role ink, so the leak is visible. `Scene.clip`/`clip_pts` on the
small scene do not transfer — a clipped small scene is REJECTED with a
ValueError rather than silently embedded unclipped. Clip the small
scene's data at compute time.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from .scene import (AngleMark, Brace, Callout, Curve, FilledCurve, Gradient,
                    Item, MathLabel, Point, RasterField, RightAngleMark,
                    Scene, Vector, XY)
from .style import Role

FRAME_CORNER_FRAC = 0.06     # corner radius as a fraction of the shorter side
_CORNER_SAMPLES = 6          # points per quarter arc


@dataclasses.dataclass(frozen=True)
class _Map:
    """The affine map: uniform scale `s` about the source origin, then
    translate so the source rect lands on the dest rect."""
    s: float
    x0: float
    y0: float
    ax: float                 # dest rect lower-left
    ay: float

    def xy(self, p: XY) -> XY:
        return (self.ax + self.s * (float(p[0]) - self.x0),
                self.ay + self.s * (float(p[1]) - self.y0))

    def pts(self, arr: np.ndarray) -> np.ndarray:
        out = np.asarray(arr, dtype=float).copy()
        out[:, 0] = self.ax + self.s * (out[:, 0] - self.x0)
        out[:, 1] = self.ay + self.s * (out[:, 1] - self.y0)
        return out


def _map_curve(it: Curve, m: _Map) -> Curve:
    return dataclasses.replace(it, pts=m.pts(it.pts))


def _map_filled(it: FilledCurve, m: _Map) -> FilledCurve:
    grad = it.gradient
    if grad is not None:
        grad = dataclasses.replace(grad, p0=m.xy(grad.p0), p1=m.xy(grad.p1))
    return dataclasses.replace(it, pts=m.pts(it.pts),
                               holes=tuple(m.pts(h) for h in it.holes),
                               gradient=grad)


def _map_vector(it: Vector, m: _Map) -> Vector:
    return dataclasses.replace(it, tail=m.xy(it.tail), tip=m.xy(it.tip))


def _map_point(it: Point, m: _Map) -> Point:
    return dataclasses.replace(it, xy=m.xy(it.xy))


def _map_label(it: MathLabel, m: _Map) -> MathLabel:
    return dataclasses.replace(it, anchor=m.xy(it.anchor))


def _map_right_angle(it: RightAngleMark, m: _Map) -> RightAngleMark:
    return dataclasses.replace(it, corner=m.xy(it.corner), size=it.size * m.s)


def _map_angle(it: AngleMark, m: _Map) -> AngleMark:
    return dataclasses.replace(it, center=m.xy(it.center), radius=it.radius * m.s)


def _map_brace(it: Brace, m: _Map) -> Brace:
    depth = it.depth * m.s if it.depth is not None else None
    return dataclasses.replace(it, p1=m.xy(it.p1), p2=m.xy(it.p2), depth=depth)


def _map_callout(it: Callout, m: _Map) -> Callout:
    return dataclasses.replace(it, anchor=m.xy(it.anchor), target=m.xy(it.target))


def _map_raster(it: RasterField, m: _Map) -> RasterField:
    x0, x1, y0, y1 = it.extent
    (nx0, ny0), (nx1, ny1) = m.xy((x0, y0)), m.xy((x1, y1))
    return dataclasses.replace(it, extent=(nx0, nx1, ny0, ny1))


_DISPATCH = {
    Curve: _map_curve,
    FilledCurve: _map_filled,
    Vector: _map_vector,
    Point: _map_point,
    MathLabel: _map_label,
    RightAngleMark: _map_right_angle,
    AngleMark: _map_angle,
    Brace: _map_brace,
    Callout: _map_callout,
    RasterField: _map_raster,
}


def _rounded_rect(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    """Closed rounded-rect polyline; corner radius FRAME_CORNER_FRAC of the
    shorter side, quarter arcs sampled at _CORNER_SAMPLES points."""
    r = FRAME_CORNER_FRAC * min(x1 - x0, y1 - y0)
    # corner centers, CCW from lower-left; each arc sweeps 90 deg CCW
    corners = [((x0 + r, y0 + r), np.pi), ((x1 - r, y0 + r), 1.5 * np.pi),
               ((x1 - r, y1 - r), 0.0), ((x0 + r, y1 - r), 0.5 * np.pi)]
    pts = []
    for (cx, cy), a0 in corners:
        th = a0 + np.linspace(0.0, 0.5 * np.pi, _CORNER_SAMPLES)
        pts.append(np.column_stack([cx + r * np.cos(th), cy + r * np.sin(th)]))
    return np.vstack(pts)


def _nearest_boundary(x0: float, y0: float, x1: float, y1: float,
                      p: XY) -> XY:
    """Nearest point of the rect BOUNDARY to p (projects outward when p is
    inside, clamps when outside)."""
    px, py = float(p[0]), float(p[1])
    cx, cy = min(max(px, x0), x1), min(max(py, y0), y1)
    if (cx, cy) != (px, py):
        return cx, cy
    # inside: push to the closest edge
    d = [(px - x0, (x0, py)), (x1 - px, (x1, py)),
         (py - y0, (px, y0)), (y1 - py, (px, y1))]
    return min(d, key=lambda e: e[0])[1]


def embed(small: Scene, *, at: XY, width: float, frame: bool = True,
          leader_to: XY | None = None, leader_pull_back_px: float = 0.0,
          key: str | None = None) -> list[Item]:
    """Map `small`'s items into host math coords; see the module docstring
    for the mapping rules. Returns items ready for `host.add(*...)`.

    `leader_pull_back_px` retracts the drawn leader head this many CANVAS
    px short of `leader_to` (drawing-only — the Vector's tip stays the
    target), so a leader can point AT a glyph without landing ON it.

    PRECONDITION: the HOST plane should be equal-aspect, or the embedded
    thumbnail is sheared — `embed` maps math coords and cannot see the
    host transform. On an anisotropic host (explicit `height_px`), do the
    px algebra at compute time: get (units/px) for both axes from
    `plots.px_units(host, width_px)` and size the small scene's ylim so
    the RENDERED height comes out right — `figures/
    saddle_node_behavior_map.py` is the worked example.
    """
    if small.xlim is None or small.ylim is None:
        raise ValueError("embed: the small scene needs explicit xlim/ylim "
                         "(the mapping needs a source rect)")
    if small.height_px is not None:
        raise ValueError("embed: height_px scenes have no math aspect; "
                         "give the small scene equal-aspect xlim/ylim")
    if small.clip is not None or small.clip_pts is not None:
        raise ValueError("embed: Scene.clip/clip_pts do not transfer (v1) "
                         "— embedding would silently draw the UNCLIPPED "
                         "content into the host. Clip the small scene's "
                         "data at compute time instead")
    sx0, sx1 = small.xlim
    sy0, sy1 = small.ylim
    dx, dy = sx1 - sx0, sy1 - sy0
    if dx <= 0 or dy <= 0:
        raise ValueError(f"embed: degenerate source rect xlim={small.xlim} "
                         f"ylim={small.ylim}")
    if width <= 0:
        raise ValueError(f"embed: width must be positive, got {width}")

    s = width / dx
    height = dy * s
    ax, ay = float(at[0]) - width / 2.0, float(at[1]) - height / 2.0
    m = _Map(s=s, x0=sx0, y0=sy0, ax=ax, ay=ay)

    out: list[Item] = []
    for it in small.items:
        mapper = _DISPATCH.get(type(it))
        if mapper is None:
            raise TypeError(
                f"embed: no mapping for item type {type(it).__name__} — "
                "add it to inset._DISPATCH (every coordinate-bearing field "
                "must be transformed)")
        out.append(mapper(it, m))

    bx0, by0, bx1, by1 = ax, ay, ax + width, ay + height
    if frame:
        out.append(Curve(_rounded_rect(bx0, by0, bx1, by1), role=Role.FRAME,
                         closed=True))
    if leader_to is not None:
        tail = _nearest_boundary(bx0, by0, bx1, by1, leader_to)
        out.append(Vector(tail, (float(leader_to[0]), float(leader_to[1])),
                          role=Role.ANNOTATION,
                          pull_back_px=leader_pull_back_px))
    if key is not None:
        out = [dataclasses.replace(it, key=key) for it in out]
    return out
