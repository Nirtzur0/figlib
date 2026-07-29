"""Compute-side scene builders: numerics that emit scene items.

Two builders, one honest output each. mapped_grid: the image of a
tensor-product grid under a plane map, with addressable cells (tracked
black cells, parity checkerboards). streamlines: level sets of a stream
function via marching squares — equal Psi-spacing makes flux tubes, so
line density encodes speed — plus RK4 field integration and stagnation
points. Style decisions stay in the figure program; these set only
role/marker defaults the caller overrides via **curve_kw.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .scene import Curve, FilledCurve
from .style import Role

ComplexMap = Callable[[np.ndarray], np.ndarray]        # C^n -> C^n, vectorized
ScalarField = Callable[[np.ndarray, np.ndarray], np.ndarray]  # psi(X, Y)
PlaneField = Callable[[np.ndarray], np.ndarray]        # (N,2) -> (N,2)


# ---------------------------------------------------------------------------
# mapped_grid


def _image_pts(f: ComplexMap, z: np.ndarray) -> np.ndarray:
    w = np.asarray(f(np.asarray(z, dtype=complex)))
    return np.column_stack([w.real, w.imag])


@dataclass
class MappedGrid:
    """Image of the grid {u_i} x {v_j} under f; cells stay addressable.

    Cell (i, j) is the image of [u_i, u_{i+1}] x [v_j, v_{j+1}]; shared
    edges are sampled identically, so cells tile exactly for injective f.
    """

    f: ComplexMap
    u: np.ndarray
    v: np.ndarray
    samples_per_line: int
    u_curves: list[Curve]   # images of the constant-u lines (v varies)
    v_curves: list[Curve]   # images of the constant-v lines (u varies)

    def cell(self, i: int, j: int) -> np.ndarray:
        """Closed boundary polygon (M, 2) of the image of grid cell (i, j)."""
        u0, u1 = self.u[i], self.u[i + 1]
        v0, v1 = self.v[j], self.v[j + 1]
        m = max(2, self.samples_per_line // 4)
        su = np.linspace(u0, u1, m)
        sv = np.linspace(v0, v1, m)
        z = np.concatenate([
            su[:-1] + 1j * v0,           # bottom: u increasing at v_j
            u1 + 1j * sv[:-1],           # right:  v increasing at u_{i+1}
            su[::-1][:-1] + 1j * v1,     # top:    u decreasing at v_{j+1}
            u0 + 1j * sv[::-1][:-1],     # left:   v decreasing at u_i
        ])
        return _image_pts(self.f, z)

    def cells(self, predicate: Callable[[int, int], bool]) -> list[tuple[int, int]]:
        return [(i, j)
                for i in range(len(self.u) - 1)
                for j in range(len(self.v) - 1)
                if predicate(i, j)]

    def cell_items(self, predicate: Callable[[int, int], bool],
                   **filled_kw) -> list[FilledCurve]:
        return [FilledCurve(self.cell(i, j), **filled_kw)
                for i, j in self.cells(predicate)]


def mapped_grid(f: ComplexMap, u: Sequence[float], v: Sequence[float],
                samples_per_line: int = 200, **curve_kw) -> MappedGrid:
    """Image of the tensor-product grid u x v under z -> f(z).

    Grid-line Curves default to Role.CONSTRUCTION (background family);
    the identity map reproduces the input grid exactly.
    """
    curve_kw.setdefault("role", Role.CONSTRUCTION)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    vd = np.linspace(v[0], v[-1], samples_per_line)
    ud = np.linspace(u[0], u[-1], samples_per_line)
    u_curves = [Curve(_image_pts(f, ui + 1j * vd), **curve_kw) for ui in u]
    v_curves = [Curve(_image_pts(f, ud + 1j * vj), **curve_kw) for vj in v]
    return MappedGrid(f, u, v, samples_per_line, u_curves, v_curves)


# ---------------------------------------------------------------------------
# marching squares


# Corner bits: a=(i,j) 1, b=(i+1,j) 2, c=(i+1,j+1) 4, d=(i,j+1) 8.
# Edges: 0 bottom (a-b), 1 right (b-c), 2 top (d-c), 3 left (a-d).
_MS_CASES: dict[int, list[tuple[int, int]]] = {
    1: [(3, 0)], 2: [(0, 1)], 3: [(3, 1)], 4: [(1, 2)],
    6: [(0, 2)], 7: [(3, 2)], 8: [(3, 2)], 9: [(0, 2)],
    11: [(1, 2)], 12: [(3, 1)], 13: [(0, 1)], 14: [(3, 0)],
}


def marching_squares(Z: np.ndarray, xs: np.ndarray, ys: np.ndarray,
                     level: float) -> list[np.ndarray]:
    """Polylines of the level set Z = level, Z[j, i] ~ f(xs[i], ys[j]).

    Linear interpolation on cell edges; saddle cells (5, 10) resolved by
    the cell-center average; cells touching a non-finite sample are
    skipped. Segments are keyed by grid-edge identity, so chains join
    exactly across cells. Closed loops repeat their first point.
    """
    ny, nx = Z.shape
    a = Z[:-1, :-1]; b = Z[:-1, 1:]; c = Z[1:, 1:]; d = Z[1:, :-1]
    finite = np.isfinite(a) & np.isfinite(b) & np.isfinite(c) & np.isfinite(d)
    with np.errstate(invalid="ignore"):
        code = ((a > level).astype(int) + 2 * (b > level) + 4 * (c > level)
                + 8 * (d > level))
    crossing = finite & (code != 0) & (code != 15)
    jj, ii = np.nonzero(crossing)

    # edge key -> interpolated crossing point (each grid edge at most once)
    pts: dict[tuple[str, int, int], tuple[float, float]] = {}

    def h_edge(j: int, i: int) -> tuple[str, int, int]:
        k = ("h", j, i)
        if k not in pts:
            t = (level - Z[j, i]) / (Z[j, i + 1] - Z[j, i])
            pts[k] = (xs[i] + t * (xs[i + 1] - xs[i]), ys[j])
        return k

    def v_edge(j: int, i: int) -> tuple[str, int, int]:
        k = ("v", j, i)
        if k not in pts:
            t = (level - Z[j, i]) / (Z[j + 1, i] - Z[j, i])
            pts[k] = (xs[i], ys[j] + t * (ys[j + 1] - ys[j]))
        return k

    adj: dict[tuple, list[tuple]] = {}
    for j, i in zip(jj, ii):
        cs = int(code[j, i])
        if cs in (5, 10):
            center_above = (a[j, i] + b[j, i] + c[j, i] + d[j, i]) / 4.0 > level
            if (cs == 5) == center_above:
                segs = [(0, 1), (2, 3)]
            else:
                segs = [(0, 3), (1, 2)]
        else:
            segs = _MS_CASES[cs]
        # lazy: only edges a segment uses get interpolated (a used edge
        # has corners on opposite sides of level, so t is never 0/0)
        def edge_key(e: int, j: int = j, i: int = i) -> tuple[str, int, int]:
            if e == 0:
                return h_edge(j, i)
            if e == 1:
                return v_edge(j, i + 1)
            if e == 2:
                return h_edge(j + 1, i)
            return v_edge(j, i)

        for e1, e2 in segs:
            k1, k2 = edge_key(e1), edge_key(e2)
            adj.setdefault(k1, []).append(k2)
            adj.setdefault(k2, []).append(k1)

    # chain segments: open paths from degree-1 edges first, then loops
    used: set[frozenset] = set()

    def walk(start: tuple) -> list[tuple]:
        line, cur = [start], start
        while True:
            nxt = next((nb for nb in adj[cur]
                        if frozenset((cur, nb)) not in used), None)
            if nxt is None:
                return line
            used.add(frozenset((cur, nxt)))
            line.append(nxt)
            cur = nxt

    lines: list[np.ndarray] = []
    for k in [k for k, ns in adj.items() if len(ns) == 1]:
        if all(frozenset((k, nb)) in used for nb in adj[k]):
            continue
        lines.append(np.array([pts[q] for q in walk(k)]))
    for k in adj:
        if any(frozenset((k, nb)) not in used for nb in adj[k]):
            lines.append(np.array([pts[q] for q in walk(k)]))
    return [ln for ln in lines if len(ln) >= 2]


# ---------------------------------------------------------------------------
# level selection


def _sample_field(scalar: ScalarField, xlim: tuple[float, float],
                  ylim: tuple[float, float], n: int, mask) -> np.ndarray:
    """Z on an n x n grid, NaN where masked (mask: bool array or callable
    (X, Y) -> bool, True = excluded) — the shared sampling convention."""
    xs = np.linspace(xlim[0], xlim[1], n)
    ys = np.linspace(ylim[0], ylim[1], n)
    X, Y = np.meshgrid(xs, ys)
    excluded = None
    if callable(mask):
        excluded = np.asarray(mask(X, Y), dtype=bool)
    elif mask is not None:
        excluded = np.asarray(mask, dtype=bool)
    if excluded is None:
        with np.errstate(all="ignore"):
            return np.asarray(scalar(X, Y), dtype=float)
    Z = np.full(X.shape, np.nan)
    keep = ~excluded
    with np.errstate(all="ignore"):
        Z[keep] = scalar(X[keep], Y[keep])
    return Z


def auto_levels(scalar: ScalarField, xlim: tuple[float, float],
                ylim: tuple[float, float], *, target_lines: int = 13,
                n: int = 256, mask=None, exclude: Sequence[float] = (),
                pct: tuple[float, float] = (2, 98)) -> list[float]:
    """Contour levels a human would have picked: robust range, round spacing.

    The field is sampled on an n x n grid (mask excludes cells, same
    convention as stream_function_lines), the range taken at the pct
    percentiles (singular spikes must not stretch the ladder), spacing
    chosen from the 1/2/5 decade ladder nearest span/target_lines, and
    levels snapped to integer multiples of the spacing. Levels within
    spacing*1e-6 of an `exclude` value are dropped (e.g. the Psi = 0
    separatrix drawn separately as the actor).
    """
    vals = _sample_field(scalar, xlim, ylim, n, mask)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return []
    lo, hi = np.percentile(vals, pct)
    span = float(hi - lo)
    if span <= 0.0 or target_lines < 1:
        return []
    raw = span / target_lines
    decade = 10.0 ** np.floor(np.log10(raw))
    ladder = np.array([1.0, 2.0, 5.0, 10.0]) * decade
    spacing = float(ladder[np.argmin(np.abs(np.log(ladder / raw)))])
    k0 = int(np.ceil(lo / spacing - 1e-9))
    k1 = int(np.floor(hi / spacing + 1e-9))
    tol = spacing * 1e-6
    return [k * spacing for k in range(k0, k1 + 1)
            if not any(abs(k * spacing - e) <= tol for e in exclude)]


# ---------------------------------------------------------------------------
# stream-function level lines


def _orient_with_flow(pts: np.ndarray, psi: ScalarField, h: float) -> np.ndarray:
    """Reverse pts if the flow v = (dPsi/dy, -dPsi/dx) opposes the tangent."""
    n = len(pts)
    for m in (n // 2, n // 4, (3 * n) // 4, 0):
        m = min(max(m, 0), n - 1)
        x, y = pts[m]
        vx = float(psi(np.float64(x), np.float64(y + h))
                   - psi(np.float64(x), np.float64(y - h))) / (2 * h)
        vy = -float(psi(np.float64(x + h), np.float64(y))
                    - psi(np.float64(x - h), np.float64(y))) / (2 * h)
        tx, ty = pts[min(m + 1, n - 1)] - pts[max(m - 1, 0)]
        dot = vx * tx + vy * ty
        if np.isfinite(dot) and dot != 0.0:
            return pts[::-1] if dot < 0 else pts
    return pts


def stream_function_lines(psi: ScalarField, xlim: tuple[float, float],
                          ylim: tuple[float, float], levels: Sequence[float],
                          n: int = 400, mask=None, arrows: tuple[float, ...] = (0.5,),
                          min_arc_frac: float = 0.03,
                          min_marker_arc: float | None = None,
                          **curve_kw) -> list[Curve]:
    """Level sets of a stream function Psi as flow-oriented Curves.

    Equal level spacing = flux tubes: line density encodes speed. mask
    (boolean (n, n) array or callable (X, Y) -> bool, True = excluded)
    drops contour segments in touched cells; Psi is sampled only outside
    the mask, so obstacle-interior singularities never blow up. Each
    polyline is oriented so v = (dPsi/dy, -dPsi/dx) points along
    increasing parameter — markers (hollow by default) point WITH the
    flow.

    Two culls, both in math units because the builder never sees a style:

    - polylines shorter than min_arc_frac of the domain diagonal are
      dropped (grid-corner stubs: marching-squares slivers that read as
      dirt, not flow);
    - direction markers are suppressed on polylines shorter than
      min_marker_arc (default 5% of the diagonal — roughly 5x a default
      arrowhead's math-space extent at column format; the honest style-
      space rule would need the arrowhead length, which is only known at
      emission).
    """
    curve_kw.setdefault("role", Role.CONTENT)
    curve_kw.setdefault("arrow_style", "hollow")
    xs = np.linspace(xlim[0], xlim[1], n)
    ys = np.linspace(ylim[0], ylim[1], n)
    Z = _sample_field(psi, xlim, ylim, n, mask)
    h = min(xs[1] - xs[0], ys[1] - ys[0])
    diag = float(np.hypot(xlim[1] - xlim[0], ylim[1] - ylim[0]))
    if min_marker_arc is None:
        min_marker_arc = 0.05 * diag

    out: list[Curve] = []
    for level in levels:
        for line in marching_squares(Z, xs, ys, float(level)):
            closed = len(line) > 3 and np.allclose(line[0], line[-1])
            pts = line[:-1] if closed else line
            if len(pts) < 2:
                continue
            seg = np.diff(np.vstack([pts, pts[:1]]) if closed else pts, axis=0)
            arc = float(np.hypot(seg[:, 0], seg[:, 1]).sum())
            if arc < min_arc_frac * diag:
                continue
            out.append(Curve(_orient_with_flow(pts, psi, h), closed=closed,
                             arrows=arrows if arc >= min_marker_arc else (),
                             **curve_kw))
    return out


# ---------------------------------------------------------------------------
# field integration and stagnation points


def integrate_streamlines(field: PlaneField, seeds: Sequence[tuple[float, float]],
                          xlim: tuple[float, float], ylim: tuple[float, float],
                          h: float = 0.02, max_len: float = 40.0,
                          stop_speed: float = 1e-8,
                          arrows: tuple[float, ...] = (0.5,),
                          **curve_kw) -> list[Curve]:
    """Unit-speed RK4 integral curves of dx/dt = field(x), one Curve per seed.

    Integrates forward AND backward, so h is arc-length step and max_len
    caps arc length per direction. Stops at domain exit, |field| <
    stop_speed (stagnation), or the cap. Markers point with the flow.
    """
    curve_kw.setdefault("role", Role.CONTENT)
    curve_kw.setdefault("arrow_style", "hollow")

    def speed_dir(x: np.ndarray) -> tuple[float, np.ndarray]:
        v = np.asarray(field(x[None, :]), dtype=float)[0]
        s = float(np.hypot(v[0], v[1]))
        return s, (v / s if s > 0 else v)

    def inside(x: np.ndarray) -> bool:
        return bool(xlim[0] <= x[0] <= xlim[1] and ylim[0] <= x[1] <= ylim[1])

    curves: list[Curve] = []
    for seed in np.atleast_2d(np.asarray(seeds, dtype=float)):
        halves: list[np.ndarray] = []
        for sign in (1.0, -1.0):
            x = seed.copy()
            pts = [x.copy()]
            for _ in range(int(max_len / h)):
                s, d1 = speed_dir(x)
                if not np.all(np.isfinite(d1)) or s < stop_speed:
                    break
                k1 = sign * d1
                k2 = sign * speed_dir(x + (h / 2) * k1)[1]
                k3 = sign * speed_dir(x + (h / 2) * k2)[1]
                k4 = sign * speed_dir(x + h * k3)[1]
                x = x + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
                if not np.all(np.isfinite(x)):
                    break
                pts.append(x.copy())
                if not inside(x):
                    break
            halves.append(np.array(pts))
        pts = np.vstack([halves[1][::-1], halves[0][1:]])
        if len(pts) >= 2:
            curves.append(Curve(pts, arrows=arrows, **curve_kw))
    return curves


def stagnation_points(field: PlaneField, xlim: tuple[float, float],
                      ylim: tuple[float, float],
                      grid_n: int = 64) -> list[tuple[float, float]]:
    """Zeros of a plane field: grid minima of |field| refined by Newton.

    Jacobian by central finite differences; only roots that converge to
    |field| < 1e-9 inside the domain are kept, deduplicated.
    """
    xs = np.linspace(xlim[0], xlim[1], grid_n)
    ys = np.linspace(ylim[0], ylim[1], grid_n)
    X, Y = np.meshgrid(xs, ys)
    P = np.column_stack([X.ravel(), Y.ravel()])
    with np.errstate(divide="ignore", invalid="ignore"):
        V = np.asarray(field(P), dtype=float)
    S = np.hypot(V[:, 0], V[:, 1]).reshape(grid_n, grid_n)
    S[~np.isfinite(S)] = np.inf
    pad = np.pad(S, 1, constant_values=np.inf)
    shifts = [(dj, di) for dj in (-1, 0, 1) for di in (-1, 0, 1) if (dj, di) != (0, 0)]
    nmin = np.min(np.stack([pad[1 + dj:grid_n + 1 + dj, 1 + di:grid_n + 1 + di]
                            for dj, di in shifts]), axis=0)
    is_min = np.isfinite(S) & (S <= nmin)

    scale = max(xlim[1] - xlim[0], ylim[1] - ylim[0])
    eps = 1e-6 * scale

    def f1(x: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.asarray(field(x[None, :]), dtype=float)[0]

    roots: list[np.ndarray] = []
    for j, i in zip(*np.nonzero(is_min)):
        x = np.array([X[j, i], Y[j, i]])
        for _ in range(40):
            v = f1(x)
            if not np.all(np.isfinite(v)) or np.hypot(v[0], v[1]) < 1e-13:
                break
            ex, ey = np.array([eps, 0.0]), np.array([0.0, eps])
            J = np.column_stack([(f1(x + ex) - f1(x - ex)) / (2 * eps),
                                 (f1(x + ey) - f1(x - ey)) / (2 * eps)])
            if not np.all(np.isfinite(J)):
                break
            try:
                d = np.linalg.solve(J, -v)
            except np.linalg.LinAlgError:
                break
            x = x + d
            if np.hypot(d[0], d[1]) < 1e-14 * scale:
                break
        v = f1(x)
        if (np.all(np.isfinite(v)) and np.hypot(v[0], v[1]) < 1e-9
                and xlim[0] <= x[0] <= xlim[1] and ylim[0] <= x[1] <= ylim[1]):
            roots.append(x)

    kept: list[np.ndarray] = []
    for r in roots:
        if all(np.hypot(*(r - k)) > 1e-6 * scale for k in kept):
            kept.append(r)
    return [(float(r[0]), float(r[1])) for r in kept]


def assert_tangents_align(curves, field: PlaneField, tol_deg: float = 1.0,
                          exclude_near: Sequence[tuple[float, float]] = (),
                          exclude_radius: float = 0.3) -> None:
    """The drawn streamlines really follow the field — for assertions().

    Signed check: wide-stencil chord tangents (pts[i+w] - pts[i-w], w a
    fraction of the polyline, so marching-squares jaggies average out)
    versus the field direction at pts[i]; the angle must stay within
    tol_deg, direction included — a level line oriented AGAINST its flow
    fails at ~180 deg, which the unsigned check would miss. Samples within
    exclude_radius of any exclude_near point are skipped (stagnation
    points and singularities turn the field direction faster than any
    chord can follow). Raises AssertionError naming the worst offender.
    """
    from .geometry import cross2d

    excl = np.asarray(exclude_near, dtype=float).reshape(-1, 2)
    worst = 0.0
    worst_at: tuple[int, float, float] | None = None
    for ci, c in enumerate(curves):
        pts = np.asarray(c.pts if isinstance(c, Curve) else c, dtype=float)
        m = len(pts)
        w = max(2, m // 16)
        if m < 2 * w + 1:
            continue
        idx = np.arange(w, m - w, max(1, (m - 2 * w) // 16))
        chords = pts[idx + w] - pts[idx - w]
        at = pts[idx]
        if len(excl):
            d = np.linalg.norm(at[:, None, :] - excl[None, :, :], axis=2)
            keep = d.min(axis=1) > exclude_radius
            chords, at = chords[keep], at[keep]
        if not len(at):
            continue
        with np.errstate(all="ignore"):
            v = np.asarray(field(at), dtype=float)
        ok = (np.isfinite(v).all(axis=1) & np.isfinite(chords).all(axis=1)
              & (np.hypot(v[:, 0], v[:, 1]) > 0)
              & (np.hypot(chords[:, 0], chords[:, 1]) > 0))
        chords, at, v = chords[ok], at[ok], v[ok]
        if not len(at):
            continue
        ang = np.degrees(np.arctan2(cross2d(chords, v),
                                    (chords * v).sum(axis=1)))
        k = int(np.argmax(np.abs(ang)))
        if abs(float(ang[k])) > worst:
            worst = abs(float(ang[k]))
            worst_at = (ci, float(at[k, 0]), float(at[k, 1]))
    if worst > tol_deg:
        ci, x, y = worst_at
        raise AssertionError(
            f"tangent deviates {worst:.2f} deg from the field (tol "
            f"{tol_deg:g}) on curve {ci} near ({x:.3g}, {y:.3g})")


# ---------------------------------------------------------------------------
# ramp_segments


def _point_at_arclen(pts: np.ndarray, s: np.ndarray, sv: float) -> np.ndarray:
    i = int(np.clip(np.searchsorted(s, sv, side="right") - 1, 0, len(pts) - 2))
    ds = s[i + 1] - s[i]
    f = (sv - s[i]) / ds if ds > 0 else 0.0
    return pts[i] + f * (pts[i + 1] - pts[i])


def ramp_segments(pts: np.ndarray, n: int = 24, *,
                  color: Callable[[float], str] | None = None,
                  opacity: Callable[[float], float] | None = None,
                  role: Role = Role.CONTENT, width_scale: float = 1.0,
                  dash: str | None = None) -> list[Curve]:
    """One polyline -> n equal-arc-length Curves whose color/opacity follow
    callables of arc-length fraction t (sampled at each segment midpoint).

    The continuous-attribute channel SVG strokes lack: time-fading
    trajectories, hue ramped along a spiral. Segments share endpoints and
    carry butt caps so translucent joints don't double-draw.
    """
    pts = np.asarray(pts, dtype=float)
    seg = np.diff(pts, axis=0)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))])
    total = float(s[-1])
    if total <= 0.0:
        raise ValueError("ramp_segments needs a curve with positive length")
    bounds = np.linspace(0.0, total, n + 1)
    out: list[Curve] = []
    for k in range(n):
        lo, hi = float(bounds[k]), float(bounds[k + 1])
        interior = pts[(s > lo) & (s < hi)]
        chunk = np.vstack([[_point_at_arclen(pts, s, lo)], interior,
                           [_point_at_arclen(pts, s, hi)]])
        t_mid = (lo + hi) / (2.0 * total)
        out.append(Curve(chunk, role=role, width_scale=width_scale, dash=dash,
                         opacity=opacity(t_mid) if opacity is not None else 1.0,
                         color=color(t_mid) if color is not None else None,
                         cap="butt"))
    return out


# ---------------------------------------------------------------------------
# domain_color


def domain_color(f: ComplexMap, xlim: tuple[float, float],
                 ylim: tuple[float, float], n: int, theme,
                 *, opacity: float = 1.0, interp: bool = True) -> "RasterField":
    """f on a complex grid -> a RasterField colored by `theme.phase`.

    A domain-coloring portrait: the plane is the domain, and every pixel
    wears the value f(z) there. It exists because the graph of a complex
    function lives in four real dimensions, so the usual moves — plot the
    curve, plot the surface — have nowhere to go. Coloring the domain trades
    the third and fourth dimensions for a color channel and gets the whole
    function on one page.

    What the reader can then read OFF the picture, without any annotation:

    - **zeros and poles, and their orders** — the hue wheel winds once per
      order, positively around a zero and negatively around a pole, so the
      order is a count of how many times the color cycles on a small loop;
    - **which is which** — the lightness trend in `theme.phase` sends zeros
      dark and poles light, so the two kinds of singularity do not have to
      be told apart by tracing a winding direction;
    - **level sets of the modulus** — the sawtooth in log2|f| draws
      |f| = 2^k as contours, for free, and their crowding is |f|'s
      steepness;
    - **conformality** — the hue and modulus contours cross at right angles
      wherever f is analytic and f' does not vanish.

    A pure producer: no roles, no annotation, no opinion about the frame.
    `interp` defaults True here (unlike the scalar RasterField) because a
    phase portrait is a continuous field sampled finely, not a lattice of
    cells the reader is meant to count.
    """
    import numpy as np

    from .scene import RasterField

    if n < 2:
        raise ValueError(f"domain_color needs n >= 2, got {n}")
    xs = np.linspace(xlim[0], xlim[1], n)
    ys = np.linspace(ylim[1], ylim[0], n)      # descending: row 0 is the top
    z = xs[None, :] + 1j * ys[:, None]
    with np.errstate(all="ignore"):
        w = np.asarray(f(z), dtype=complex)
    rgb = theme.phase(np.angle(w), np.abs(w))
    return RasterField(rgb, extent=(float(xlim[0]), float(xlim[1]),
                                    float(ylim[0]), float(ylim[1])),
                       opacity=opacity, interp=interp)
