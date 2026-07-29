"""Data plots as scene items: scales, axes, series, bands, phase lines.

A producer module in the sense of surface3d — pure functions that EMIT
scene primitives. Nothing here renders, and nothing names a color or a
stroke width: style stays at the call site as Roles, dash names and
marker shapes, so a plot inherits the theme and all three gates for
free.

The one structural idea: **scales are compute-side**. A Scale maps data
into *axis coordinates* before anything enters a Scene, so a log axis is
ordinary geometry, ticks are ordinary Curves, and layout, extents and
the mechanical gate work unchanged. There is no second coordinate system
and no renderer-side transform to keep in sync.

Sizes (tick length, marker radius) are in MATH units, the only frame a
scene item has. `px_units` converts a panel's canvas pixels into math
units for the cases where a marker must look round, not merely be round.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .gates import Checks
from .layout import Transform
from .scene import Curve, FilledCurve, Item, MathLabel, Scene, Vector
from .style import Role

ArrayLike = Sequence[float] | np.ndarray


# ---------------------------------------------------------------------------
# scales


@dataclass(frozen=True)
class Ticks:
    """Ticks in both frames: `values` are data, `positions` are axis
    coordinates (= scale.fwd(values)); `minor` are unlabeled positions."""

    values: np.ndarray
    positions: np.ndarray
    labels: tuple[str, ...]
    minor: np.ndarray = field(default_factory=lambda: np.empty(0))


def nice_step(span: float, target: int = 5) -> float:
    """Clean tick step: 1, 2 or 5 times a power of ten — the smallest such
    step that cuts `span` into at most `target` intervals. Rounding the
    step UP, never the count, is what keeps labels at round numbers; the
    realized count lands in [target/2.5, target]."""
    if not math.isfinite(span) or span <= 0:
        raise ValueError(f"tick step needs a positive finite span, got {span!r}")
    raw = span / max(target, 1)
    mag = 10.0 ** math.floor(math.log10(raw))
    for m in (1.0, 2.0, 5.0):
        if raw <= m * mag * (1.0 + 1e-9):
            return m * mag
    return 10.0 * mag


def _fmt_decimal(v: float, step: float) -> str:
    """Tick label carrying exactly the decimals the step demands."""
    dec = max(0, int(math.ceil(-math.log10(step) - 1e-9)))
    if abs(v) < step * 1e-6:
        v = 0.0
    return f"{v:.{dec}f}"


@dataclass(frozen=True)
class Scale:
    """Data -> axis coordinates, applied before anything enters a Scene."""

    lo: float
    hi: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.lo) and math.isfinite(self.hi)):
            raise ValueError(f"scale bounds must be finite, got ({self.lo}, {self.hi})")
        if self.hi <= self.lo:
            raise ValueError(f"scale needs lo < hi, got ({self.lo}, {self.hi})")

    def fwd(self, v: ArrayLike) -> np.ndarray:
        raise NotImplementedError

    def inv(self, u: ArrayLike) -> np.ndarray:
        raise NotImplementedError

    @property
    def range(self) -> tuple[float, float]:
        """The axis-coordinate interval the axis occupies."""
        a, b = self.fwd([self.lo, self.hi])
        return (float(a), float(b))

    @property
    def span(self) -> float:
        a, b = self.range
        return b - a

    def ticks(self, target: int = 5, *, minor: bool = False,
              step: float | None = None) -> Ticks:
        raise NotImplementedError


@dataclass(frozen=True)
class Linear(Scale):
    def fwd(self, v: ArrayLike) -> np.ndarray:
        return np.asarray(v, dtype=float)

    def inv(self, u: ArrayLike) -> np.ndarray:
        return np.asarray(u, dtype=float)

    def ticks(self, target: int = 5, *, minor: bool = False,
              step: float | None = None) -> Ticks:
        h = nice_step(self.hi - self.lo, target) if step is None else float(step)
        if h <= 0:
            raise ValueError(f"tick step must be positive, got {h}")
        k0 = int(math.ceil(self.lo / h - 1e-9))
        k1 = int(math.floor(self.hi / h + 1e-9))
        # round away the 1e-17 debris of k*h so labels and assertions agree
        vals = np.round(np.arange(k0, k1 + 1, dtype=float) * h, 12)
        mins = np.empty(0)
        if minor:
            m = np.arange(k0 - 1, k1 + 1, dtype=float) * h + h / 2.0
            mins = np.round(m[(m >= self.lo) & (m <= self.hi)], 12)
        return Ticks(values=vals, positions=vals.copy(),
                     labels=tuple(_fmt_decimal(float(v), h) for v in vals),
                     minor=mins)


@dataclass(frozen=True)
class Log10(Scale):
    """Decades. Nonpositive data is an error, never a silently dropped point."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.lo <= 0.0:
            raise ValueError(
                f"log10 scale needs lo > 0, got lo={self.lo} — a log axis has "
                f"no room for zero or negative data")

    def fwd(self, v: ArrayLike) -> np.ndarray:
        a = np.asarray(v, dtype=float)
        bad = ~(a > 0.0)
        if np.any(bad):
            raise ValueError(
                f"log10 scale: {int(bad.sum())} of {a.size} values are "
                f"nonpositive (min {np.nanmin(a):.6g}) — filter or shift at the "
                f"call site; the axis must not hide them")
        return np.log10(a)

    def inv(self, u: ArrayLike) -> np.ndarray:
        return np.power(10.0, np.asarray(u, dtype=float))

    def ticks(self, target: int = 5, *, minor: bool = False,
              step: float | None = None) -> Ticks:
        if step is not None:
            raise ValueError("log10 ticks sit at decades; `step` does not apply")
        k0 = int(math.ceil(math.log10(self.lo) - 1e-9))
        k1 = int(math.floor(math.log10(self.hi) + 1e-9))
        decades = np.arange(k0, k1 + 1, dtype=float)
        mins = np.empty(0)
        if minor:
            m = np.array([d + math.log10(j)
                          for d in range(k0 - 1, k1 + 1) for j in range(2, 10)])
            u0, u1 = self.range
            mins = m[(m >= u0) & (m <= u1)]
        return Ticks(values=np.power(10.0, decades), positions=decades,
                     labels=tuple(f"10^{{{int(d)}}}" for d in decades),
                     minor=mins)


def linear(lo: float, hi: float) -> Linear:
    return Linear(float(lo), float(hi))


def log10(lo: float, hi: float) -> Log10:
    return Log10(float(lo), float(hi))


# ---------------------------------------------------------------------------
# axis


def _pt(along: float, cross: float, orient: str) -> tuple[float, float]:
    return (along, cross) if orient == "x" else (cross, along)


def _tick_label_anchor(orient: str, side: int, pad: float
                       ) -> tuple[str, str, tuple[float, float]]:
    """(ha, va, offset_px) for a tick label on the `side` of the axis.
    Canvas +y points DOWN, hence the sign flip on the x orientation."""
    if orient == "x":
        return ("center", "top", (0.0, pad)) if side < 0 else \
               ("center", "bottom", (0.0, -pad))
    return ("right", "center", (-pad, 0.0)) if side < 0 else \
           ("left", "center", (pad, 0.0))


def axis(scale: Scale, *, orient: str = "x", at: float = 0.0, side: int = -1,
         ticks: Ticks | None = None, tick_len: float | None = None,
         minor_frac: float = 0.55, tick_labels: bool = True,
         skip: Sequence[float] = (), label: str | None = None,
         label_at: str = "end", arrow: bool = True,
         extend: tuple[float, float] = (0.0, 0.0),
         grid_to: float | None = None, role: Role = Role.ANNOTATION,
         tick_role: Role | None = None, label_role: Role | None = None,
         grid_role: Role = Role.FRAME, size_pt: float | None = None,
         tick_pad_px: float = 5.0, label_pad_px: float = 8.0,
         dash: str | None = None) -> list[Item]:
    """Spine + ticks + tick labels (+ grid, + axis label) as scene items.

    `at` is the cross-axis coordinate of the spine, so the axis is
    *placeable* — a phase portrait wants it through the origin, not welded
    to the frame — and `side` (-1 below/left, +1 above/right) puts the
    ticks and their labels. `tick_len` and `grid_to` are cross-axis math
    units; the tick default assumes roughly equal aspect. `extend`
    lengthens the spine past the scale's range at (start, end), leaving
    room for the arrowhead. `label_at` is "end" (the math-axis look: the
    name sits past the arrow) or "center" (the data-plot look: centered,
    rotated on a y axis).

    Everything is returned, nothing is added — the call site keeps control
    of z-order and of which scene it lands in.
    """
    if orient not in ("x", "y"):
        raise ValueError(f"orient must be 'x' or 'y', got {orient!r}")
    if side not in (-1, 1):
        raise ValueError(f"side must be -1 or +1, got {side!r}")
    if label_at not in ("end", "center"):
        raise ValueError(f"label_at must be 'end' or 'center', got {label_at!r}")
    tick_role = role if tick_role is None else tick_role
    label_role = role if label_role is None else label_role
    u0, u1 = scale.range
    tl = tick_len if tick_len is not None else 0.018 * (u1 - u0)
    tk = scale.ticks() if ticks is None else ticks

    items: list[Item] = [Curve(
        np.array([_pt(u0 - extend[0], at, orient), _pt(u1 + extend[1], at, orient)]),
        role=role, dash=dash, arrows=(1.0,) if arrow else ())]

    for p in tk.minor:
        items.append(Curve(
            np.array([_pt(float(p), at, orient),
                      _pt(float(p), at + side * tl * minor_frac, orient)]),
            role=tick_role, dash=dash))

    ha, va, off = _tick_label_anchor(orient, side, tick_pad_px)
    for p, v, text in zip(tk.positions, tk.values, tk.labels):
        p = float(p)
        if grid_to is not None:
            items.append(Curve(np.array([_pt(p, at, orient), _pt(p, grid_to, orient)]),
                               role=grid_role))
        items.append(Curve(np.array([_pt(p, at, orient),
                                     _pt(p, at + side * tl, orient)]),
                           role=tick_role, dash=dash))
        if tick_labels and not any(abs(float(v) - s) <= 1e-9 * max(1.0, abs(s))
                                   for s in skip):
            items.append(MathLabel(text, _pt(p, at + side * tl, orient),
                                   role=label_role, ha=ha, va=va,
                                   offset_px=off, size_pt=size_pt))

    if label is not None:
        items.append(_axis_label(label, scale, orient, at, side, extend,
                                 label_at, label_role, size_pt, label_pad_px))
    return items


def _axis_label(latex: str, scale: Scale, orient: str, at: float, side: int,
                extend: tuple[float, float], label_at: str, role: Role,
                size_pt: float | None, pad: float) -> MathLabel:
    u0, u1 = scale.range
    if label_at == "end":
        end = _pt(u1 + extend[1], at, orient)
        if orient == "x":
            return MathLabel(latex, end, role=role, ha="left", va="center",
                             offset_px=(pad, 0.0), size_pt=size_pt)
        return MathLabel(latex, end, role=role, ha="center", va="bottom",
                         offset_px=(0.0, -pad), size_pt=size_pt)
    mid = _pt(0.5 * (u0 + u1), at, orient)
    if orient == "x":
        return MathLabel(latex, mid, role=role, ha="center",
                         va="top" if side < 0 else "bottom",
                         offset_px=(0.0, side * -pad if side > 0 else pad),
                         size_pt=size_pt)
    return MathLabel(latex, mid, role=role, ha="center", va="bottom",
                     offset_px=(side * pad, 0.0), angle_deg=90.0, size_pt=size_pt)


# ---------------------------------------------------------------------------
# series, markers, bands, histograms


# Unit marker shapes have circumradius 1; these factors equalize AREA, so
# shape stays a pure identity channel — a channel whose levels differ in
# salience is quietly encoding a second thing.
_SHAPE_AREA_SCALE = {"circle": 1.0, "square": 0.886, "diamond": 1.253,
                     "triangle": 1.555}


def _unit_shape(shape: str, n_circle: int) -> np.ndarray:
    if shape == "circle":
        th = np.linspace(0.0, 2 * np.pi, n_circle, endpoint=False)
        return np.column_stack([np.cos(th), np.sin(th)])
    if shape == "square":
        return np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    if shape == "diamond":
        return np.array([[0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    if shape == "triangle":
        th = np.pi / 2 + np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3])
        return np.column_stack([np.cos(th), np.sin(th)])
    raise ValueError(f"unknown marker shape {shape!r} — circle | square | "
                     f"diamond | triangle")


def _radii(size: float | tuple[float, float]) -> tuple[float, float]:
    if isinstance(size, (int, float)):
        return (float(size), float(size))
    rx, ry = size
    return (float(rx), float(ry))


def markers(x: ArrayLike, y: ArrayLike, shape: str = "circle", *,
            size: float | tuple[float, float] = 0.03, filled: bool = True,
            role: Role = Role.CONTENT, color: str | None = None,
            n_circle: int = 24, width_scale: float = 1.0) -> list[Item]:
    """Point markers as closed geometry, one item per point.

    Shape is an identity channel orthogonal to hue and dash, so a
    monochrome print keeps every distinction the color print had. Filled
    means attained, hollow means excluded (grammar.md); shapes are drawn
    at equal area and sized in math units, so `size` obeys the panel's
    scales — pass (rx, ry) where the aspect is not 1.

    Hollow markers do NOT knock ink out from under themselves (the scene
    has no paper-filled polygon primitive); break the line at the marker,
    as `phase_line` does, wherever one would run through.
    """
    rx, ry = _radii(size)
    xs = np.asarray(x, dtype=float).ravel()
    ys = np.asarray(y, dtype=float).ravel()
    if xs.shape != ys.shape:
        raise ValueError(f"marker x/y length mismatch: {xs.shape} vs {ys.shape}")
    if shape == "cross":
        # the pole glyph. Stroke-only by nature — a x has no interior —
        # so filled would silently mean nothing; diagonals scaled so
        # tip-to-tip span equals the circle's diameter at the same size
        if filled:
            raise ValueError("cross markers are stroke-only — pass filled=False")
        c = 1.0 / math.sqrt(2.0)
        out_x: list[Item] = []
        for cx, cy in zip(xs, ys):
            for sy in (1.0, -1.0):
                out_x.append(Curve(
                    np.array([[cx - rx * c, cy - sy * ry * c],
                              [cx + rx * c, cy + sy * ry * c]]),
                    role=role, color=color, width_scale=width_scale))
        return out_x
    unit = _unit_shape(shape, n_circle) * _SHAPE_AREA_SCALE.get(shape, 1.0)
    out: list[Item] = []
    for cx, cy in zip(xs, ys):
        pts = np.column_stack([cx + rx * unit[:, 0], cy + ry * unit[:, 1]])
        out.append(FilledCurve(pts, role=role, opacity=1.0, outline=True,
                               color=color) if filled
                   else Curve(pts, closed=True, role=role, color=color,
                              width_scale=width_scale))
    return out


def stems(x: ArrayLike, y: ArrayLike, *, baseline: float = 0.0,
          marker: str | None = "circle", filled: bool = True,
          size: float | tuple[float, float] = 0.03,
          xscale: Scale | None = None, yscale: Scale | None = None,
          role: Role = Role.CONTENT, color: str | None = None,
          width_scale: float = 1.0) -> list[Item]:
    """A discrete sequence x[n]: one baseline-to-sample stem plus a marker
    head per sample.

    This is the visual TYPE for sampled data — a measure (delta train) is
    `impulses`, a continuous function is `series`; keeping the three
    distinct is the domain's load-bearing convention. The stem stops one
    marker-radius short of a hollow head so the head reads hollow (the
    phase_line trick); a sample shorter than that gap keeps its marker
    and drops the stem. Scales are applied here, once, baseline included.
    """
    xs = np.asarray(x, dtype=float).ravel()
    ys = np.asarray(y, dtype=float).ravel()
    if xs.shape != ys.shape:
        raise ValueError(f"stems x/y length mismatch: {xs.shape} vs {ys.shape}")
    ux = xscale.fwd(xs) if xscale is not None else xs
    uy = yscale.fwd(ys) if yscale is not None else ys
    base = float(yscale.fwd([baseline])[0]) if yscale is not None else float(baseline)
    _, ry = _radii(size)
    gap = ry if (marker is not None and not filled) else 0.0
    items: list[Item] = []
    for cx, cy in zip(ux, uy):
        s = 1.0 if cy >= base else -1.0
        stop = cy - s * gap
        if s * (stop - base) > 1e-12:
            items.append(Curve(np.array([[cx, base], [cx, stop]]), role=role,
                               color=color, width_scale=width_scale))
    if marker is not None:
        items += markers(ux, uy, marker, size=size, filled=filled, role=role,
                         color=color, width_scale=width_scale)
    return items


def impulses(x: ArrayLike, weights: ArrayLike, *, baseline: float = 0.0,
             xscale: Scale | None = None, yscale: Scale | None = None,
             role: Role = Role.CONTENT,
             width_scale: float = 1.0) -> list[Vector]:
    """Bracewell's delta convention: an arrow whose HEIGHT is the weight.

    A distribution has no graph, so what gets drawn is the measure — an
    impulse of weight 2 is an arrow twice as tall, negative weight points
    down. The arrowhead is what distinguishes it from a stem at a glance;
    that type distinction (arrow = measure, lollipop = sample) is the
    point of having two producers. `weights` are axis-coordinate lengths:
    a weight is not a data value and does not pass through a scale;
    `baseline` does.
    """
    xs = np.asarray(x, dtype=float).ravel()
    ws = np.asarray(weights, dtype=float).ravel()
    if xs.shape != ws.shape:
        raise ValueError(
            f"impulses x/weights length mismatch: {xs.shape} vs {ws.shape}")
    ux = xscale.fwd(xs) if xscale is not None else xs
    base = float(yscale.fwd([baseline])[0]) if yscale is not None else float(baseline)
    return [Vector((float(cx), base), (float(cx), base + float(w)),
                   role=role, width_scale=width_scale)
            for cx, w in zip(ux, ws)]


def series(x: ArrayLike, y: ArrayLike, *, xscale: Scale | None = None,
           yscale: Scale | None = None, line: bool = True,
           marker: str | None = None,
           marker_size: float | tuple[float, float] = 0.03,
           marker_every: int = 1, marker_filled: bool = True,
           marker_role: Role | None = None, role: Role = Role.CONTENT,
           color: str | None = None, **curve_kw) -> list[Item]:
    """One data series: a Curve in axis coordinates, plus optional markers.

    Scales are applied here, once; everything downstream is geometry.
    """
    xs = np.asarray(x, dtype=float).ravel()
    ys = np.asarray(y, dtype=float).ravel()
    if xs.shape != ys.shape:
        raise ValueError(f"series x/y length mismatch: {xs.shape} vs {ys.shape}")
    if xs.size == 0:
        raise ValueError("series needs at least one point")
    ux = xscale.fwd(xs) if xscale is not None else xs
    uy = yscale.fwd(ys) if yscale is not None else ys
    items: list[Item] = []
    if line:
        if ux.size < 2:
            raise ValueError("a line series needs at least two points")
        items.append(Curve(np.column_stack([ux, uy]), role=role, color=color,
                           **curve_kw))
    if marker is not None:
        if marker_every < 1:
            raise ValueError(f"marker_every must be >= 1, got {marker_every}")
        items += markers(ux[::marker_every], uy[::marker_every], marker,
                         size=marker_size, filled=marker_filled,
                         role=role if marker_role is None else marker_role,
                         color=color)
    return items


def band(x: ArrayLike, y_lo: ArrayLike, y_hi: ArrayLike, *,
         xscale: Scale | None = None, yscale: Scale | None = None,
         role: Role = Role.MUTED, opacity: float = 0.18,
         outline: bool = False, **filled_kw) -> FilledCurve:
    """The region between two curves over a shared x — confidence
    intervals, ensemble envelopes. Closed by walking out along y_lo and
    back along y_hi, so the polygon is simple whenever y_lo <= y_hi."""
    xs = np.asarray(x, dtype=float).ravel()
    lo = np.asarray(y_lo, dtype=float).ravel()
    hi = np.asarray(y_hi, dtype=float).ravel()
    if not (xs.shape == lo.shape == hi.shape):
        raise ValueError(f"band needs equal lengths, got {xs.shape}, "
                         f"{lo.shape}, {hi.shape}")
    if xs.size < 2:
        raise ValueError("band needs at least two x samples")
    ux = xscale.fwd(xs) if xscale is not None else xs
    ulo = yscale.fwd(lo) if yscale is not None else lo
    uhi = yscale.fwd(hi) if yscale is not None else hi
    pts = np.vstack([np.column_stack([ux, ulo]),
                     np.column_stack([ux[::-1], uhi[::-1]])])
    return FilledCurve(pts, role=role, opacity=opacity, outline=outline,
                       **filled_kw)


def histogram(counts: ArrayLike, edges: ArrayLike, *, baseline: float = 0.0,
              xscale: Scale | None = None, yscale: Scale | None = None,
              per_bin: bool = False, role: Role = Role.CONTENT,
              opacity: float = 0.22, outline: bool = True,
              **filled_kw) -> list[FilledCurve]:
    """Precomputed counts + bin edges -> step-outline FilledCurve(s).

    The counts are the caller's: binning is numerics, and hiding a bin
    width inside the drawing layer hides a modelling choice. One polygon
    by default; one per bin with `per_bin`, so bins can carry their own
    color.
    """
    c = np.asarray(counts, dtype=float).ravel()
    e = np.asarray(edges, dtype=float).ravel()
    if e.size != c.size + 1:
        raise ValueError(f"histogram needs len(edges) == len(counts) + 1, "
                         f"got {e.size} and {c.size}")
    if np.any(np.diff(e) <= 0):
        raise ValueError("histogram edges must be strictly increasing")
    ue = xscale.fwd(e) if xscale is not None else e
    top = yscale.fwd(c) if yscale is not None else c
    base = float(yscale.fwd([baseline])[0]) if yscale is not None else float(baseline)

    if per_bin:
        return [FilledCurve(np.array([[ue[i], base], [ue[i], top[i]],
                                      [ue[i + 1], top[i]], [ue[i + 1], base]]),
                            role=role, opacity=opacity, outline=outline,
                            **filled_kw)
                for i in range(c.size)]
    pts = [[ue[0], base]]
    for i in range(c.size):
        pts += [[ue[i], top[i]], [ue[i + 1], top[i]]]
    pts.append([ue[-1], base])
    return [FilledCurve(np.array(pts), role=role, opacity=opacity,
                        outline=outline, **filled_kw)]


# ---------------------------------------------------------------------------
# the phase line: 1D dynamics, xdot = f(x)


@dataclass(frozen=True)
class FixedPoint:
    """A root of f. `stable` is True (attracting), False (repelling), or
    None for a half-stable root — f'(x*) = 0, attracting from one side
    and repelling from the other."""

    x: float
    stable: bool | None


def flow_intervals(f: Callable[[np.ndarray], np.ndarray],
                   fixed_xs: Sequence[float], xlim: tuple[float, float], *,
                   frac: float = 0.5) -> np.ndarray:
    """(M, 2) array of (tail, tip): one flow arrow per open interval
    between consecutive fixed points, plus the two unbounded ends.

    Each arrow is centered on its interval, spans `frac` of it, and points
    along sign f(midpoint) — so its direction is a computed quantity the
    numerical gate can check against f, not a drawing decision.
    """
    bounds = [float(xlim[0])] + sorted(float(v) for v in fixed_xs) + [float(xlim[1])]
    out: list[tuple[float, float]] = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a <= 0:
            continue
        mid = 0.5 * (a + b)
        s = float(np.sign(np.asarray(f(np.array([mid])), dtype=float)[0]))
        if s == 0.0:
            continue
        half = 0.5 * frac * (b - a)
        out.append((mid - s * half, mid + s * half))
    return np.array(out, dtype=float).reshape(-1, 2)


def _half_disc(cx: float, cy: float, rx: float, ry: float, side: int,
               n: int = 25) -> np.ndarray:
    """Closed half-disc covering the `side` (-1 left, +1 right) half. Odd
    n, so the extreme point of the arc is a vertex and the drawn half is
    exactly a half."""
    th0 = np.pi / 2 if side < 0 else -np.pi / 2
    th = th0 + np.linspace(0.0, np.pi, n)
    return np.column_stack([cx + rx * np.cos(th), cy + ry * np.sin(th)])


def _attracting_side(x: float, arrows: np.ndarray) -> int:
    """Which half of a half-stable dot to ink: the side the flow arrives
    from. Read off the nearest arrow to its left — rightward flow there
    means the left side attracts."""
    left = [row for row in np.atleast_2d(arrows) if max(row) <= x + 1e-12]
    if not left:
        return -1
    row = max(left, key=lambda r: max(r))
    return -1 if row[1] > row[0] else +1


def phase_line(fixed: Sequence[FixedPoint], arrows: np.ndarray, *,
               xlim: tuple[float, float], y: float = 0.0,
               dot: float | tuple[float, float] = 0.05,
               spine: bool = True, spine_arrow: bool = False,
               spine_role: Role = Role.ANNOTATION,
               dot_role: Role = Role.CONTENT,
               arrow_role: Role = Role.ACCENT1,
               arrow_width: float = 1.0) -> list[Item]:
    """The phase line of xdot = f(x): state axis, fixed points, flow.

    Filled dot = stable, hollow = unstable, half-filled = half-stable with
    the attracting half inked — the convention IS the content here, so it
    is drawn rather than captioned. The spine is broken at every fixed
    point, which is what lets a hollow dot read as hollow without a
    paper-filled polygon in the scene.

    `arrows` is the (M, 2) array from `flow_intervals`; passing it in
    rather than recomputing keeps the drawn arrows and the asserted
    arrows the same array.
    """
    rx, ry = _radii(dot)
    xs = sorted(float(p.x) for p in fixed)
    items: list[Item] = []

    if spine:
        cuts = [float(xlim[0])]
        for xv in xs:
            cuts += [xv - 1.05 * rx, xv + 1.05 * rx]
        cuts.append(float(xlim[1]))
        segs = [(cuts[i], cuts[i + 1]) for i in range(0, len(cuts), 2)]
        for k, (a, b) in enumerate(segs):
            if b - a <= 0:
                continue
            last = k == len(segs) - 1
            items.append(Curve(np.array([[a, y], [b, y]]), role=spine_role,
                               arrows=(1.0,) if (spine_arrow and last) else ()))

    for a, b in np.atleast_2d(arrows):
        items.append(Vector((float(a), y), (float(b), y), role=arrow_role,
                            width_scale=arrow_width))

    for p in fixed:
        if p.stable is None:
            items.append(FilledCurve(
                _half_disc(float(p.x), y, rx, ry, _attracting_side(float(p.x), arrows)),
                role=dot_role, opacity=1.0, outline=False))
        items += markers([p.x], [y], "circle", size=(rx, ry),
                         filled=p.stable is True, role=dot_role)
    return items


# ---------------------------------------------------------------------------
# honesty checks, and the px bridge they share with markers


def px_units(scene: Scene, width_px: float) -> tuple[float, float]:
    """Math units per canvas pixel, (x, y), for a scene rendered at
    `width_px` — the bridge for sizing a marker in pixels. Built from the
    very Transform the renderer will use, so it is exact, not an estimate.
    A scene with explicit xlim/ylim can be measured before it has items."""
    t = Transform(scene, width_px=width_px)
    return (1.0 / t.scale_x, 1.0 / t.scale_y)


def tick_honesty(checks: Checks, scale: Scale, data: ArrayLike, *,
                 name: str = "axis", min_span_frac: float = 0.40,
                 ticks: Ticks | None = None) -> None:
    """Assert that an axis is not lying about the data it carries.

    The mechanical ways an axis lies:
      * data outside the axis range — points silently clipped away;
      * data occupying a small fraction of the range — padding that
        flattens the very variation the plot exists to show (floor 40% by
        default; lower it only when the empty region is itself content,
        and say why at the call site);
      * a log axis over nonpositive data; ticks outside the range; fewer
        than two ticks, which leaves the scale unreadable.

    Accumulates into a gates.Checks, so one gate run reports every defect
    rather than the first.
    """
    a = np.asarray(data, dtype=float).ravel()
    checks.check(a.size > 0, f"{name}: no data to check")
    if a.size == 0:
        return
    checks.check(bool(np.all(np.isfinite(a))),
                 f"{name}: {int((~np.isfinite(a)).sum())} non-finite data values")
    a = a[np.isfinite(a)]
    if a.size == 0:
        return
    lo, hi = float(a.min()), float(a.max())
    tol = 1e-9 * max(1.0, abs(scale.hi - scale.lo))
    checks.check(lo >= scale.lo - tol,
                 f"{name}: data min {lo:.6g} below axis lo {scale.lo:.6g}")
    checks.check(hi <= scale.hi + tol,
                 f"{name}: data max {hi:.6g} above axis hi {scale.hi:.6g}")
    if isinstance(scale, Log10) and lo <= 0.0:
        checks.check(False,
                     f"{name}: log axis over nonpositive data (min {lo:.6g})")
        return
    frac = abs(float(scale.fwd([hi])[0] - scale.fwd([lo])[0])) / scale.span
    checks.check(frac >= min_span_frac,
                 f"{name}: data spans {frac:.0%} of the axis range "
                 f"(floor {min_span_frac:.0%}) — the axis is mostly padding")
    tk = scale.ticks() if ticks is None else ticks
    checks.check(tk.values.size >= 2,
                 f"{name}: {tk.values.size} major tick(s) — an axis with fewer "
                 f"than two ticks has no readable scale")
    u0, u1 = scale.range
    outside = int(np.sum((tk.positions < u0 - tol) | (tk.positions > u1 + tol)))
    checks.check(outside == 0, f"{name}: {outside} tick(s) outside the axis range")
