"""Orthographic 3D: projection, painter's algorithm, Lambertian shading.

Produces ordinary 2D scene items (FilledCurve quads, Curve segments)
sorted far-to-near, so the whole 2D pipeline — gates included — applies
unchanged to 3D figures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .geometry import cross2d
from .scene import Curve, FilledCurve, MathLabel, Vector
from .style import DEFAULT_STYLE, Role


@dataclass(frozen=True)
class Camera:
    azim_deg: float = -35.0
    elev_deg: float = 32.0

    def axes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a = math.radians(self.azim_deg)
        e = math.radians(self.elev_deg)
        right = np.array([-math.sin(a), math.cos(a), 0.0])
        up = np.array([-math.cos(a) * math.sin(e), -math.sin(a) * math.sin(e), math.cos(e)])
        toward = np.array([math.cos(a) * math.cos(e), math.sin(a) * math.cos(e), math.sin(e)])
        return right, up, toward


def project(pts3: np.ndarray, cam: Camera) -> tuple[np.ndarray, np.ndarray]:
    """(N,3) world -> ((N,2) screen, (N,) depth toward viewer)."""
    right, up, toward = cam.axes()
    return np.column_stack([pts3 @ right, pts3 @ up]), pts3 @ toward


def _mix(hex_a: str, hex_b: str, t: float) -> str:
    a = [int(hex_a[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(hex_b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(u + t * (v - u)):02x}" for u, v in zip(a, b))


LIGHT = np.array([-0.35, 0.25, 0.9])
LIGHT_DIR = LIGHT / np.linalg.norm(LIGHT)


def surface_items(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, cam: Camera,
                  dark: str = "#5f6f8f", lite: str = "#f4f6fa",
                  edge: str = "#6a6f7a", edge_width: float = 0.35,
                  mask: np.ndarray | None = None,
                  shade=None) -> list[tuple[float, FilledCurve]]:
    """Shaded quads of the graph surface, each tagged with its depth.

    mask (same shape as Z, boolean): quads with any masked corner are
    skipped (e.g. clipped pole chimneys).
    """
    items: list[tuple[float, FilledCurve]] = []
    n, m = Z.shape
    for i in range(n - 1):
        for j in range(m - 1):
            if mask is not None and (mask[i, j] or mask[i + 1, j] or mask[i, j + 1] or mask[i + 1, j + 1]):
                continue
            quad3 = np.array([
                [X[i, j], Y[i, j], Z[i, j]],
                [X[i + 1, j], Y[i + 1, j], Z[i + 1, j]],
                [X[i + 1, j + 1], Y[i + 1, j + 1], Z[i + 1, j + 1]],
                [X[i, j + 1], Y[i, j + 1], Z[i, j + 1]],
            ])
            normal = np.cross(quad3[2] - quad3[0], quad3[3] - quad3[1])
            nn = np.linalg.norm(normal)
            if nn == 0:
                continue
            normal = normal / nn if normal[2] >= 0 else -normal / nn
            t = 0.25 + 0.75 * max(0.0, float(normal @ LIGHT_DIR))
            fill = shade(t) if shade is not None else _mix(dark, lite, t)
            pts2, depth = project(quad3, cam)
            items.append((float(depth.mean()), FilledCurve(
                pts2, role=Role.CONTENT, opacity=1.0, outline=False,
                color=fill, edge_color=edge,
                edge_width=edge_width)))
    return items


def polyline_items(pts3: np.ndarray, cam: Camera, depth_bias: float = 0.04,
                   **curve_kwargs) -> list[tuple[float, Curve]]:
    """A 3D polyline as per-segment 2D curves, depth-sorted with the quads.

    depth_bias lifts segments toward the viewer so a trace lying ON the
    surface wins the paint order against its own supporting quads.
    """
    pts2, depth = project(pts3, cam)
    out = []
    for i in range(len(pts3) - 1):
        seg = pts2[i:i + 2]
        out.append((float(depth[i:i + 2].mean()) + depth_bias, Curve(seg, **curve_kwargs)))
    return out


def compose(*groups: list[tuple[float, object]]) -> list:
    """Merge depth-tagged item groups, far first (painter's algorithm)."""
    merged = [pair for g in groups for pair in g]
    merged.sort(key=lambda p: p[0])
    return [item for _, item in merged]


@dataclass
class _DepthBuffer:
    """Screen-space nearest-depth raster; -inf where no surface projects."""
    buf: np.ndarray                # (ny, nx)
    origin: tuple[float, float]    # screen coords of cell (0, 0) corner
    cell: float

    def lookup(self, pts2: np.ndarray) -> np.ndarray:
        ix = np.clip(((pts2[:, 0] - self.origin[0]) / self.cell).astype(int),
                     0, self.buf.shape[1] - 1)
        iy = np.clip(((pts2[:, 1] - self.origin[1]) / self.cell).astype(int),
                     0, self.buf.shape[0] - 1)
        return self.buf[iy, ix]


def _depth_buffer(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, cam: Camera,
                  grid: int = 256) -> _DepthBuffer:
    """Rasterize the projected quads' depths (two triangles each, barycentric
    interpolation at cell centers, nearest wins). O(quads x cells-per-quad)."""
    n, m = Z.shape
    pts3 = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    pts2, depth = project(pts3, cam)
    P = pts2.reshape(n, m, 2)
    D = depth.reshape(n, m)
    x0, y0 = pts2.min(axis=0)
    x1, y1 = pts2.max(axis=0)
    cell = max(x1 - x0, y1 - y0, 1e-12) / grid
    nx = int(np.ceil((x1 - x0) / cell)) + 1
    ny = int(np.ceil((y1 - y0) / cell)) + 1
    buf = np.full((ny, nx), -np.inf)

    def raster_tri(p0, p1, p2, z0, z1, z2):
        e1, e2 = p1 - p0, p2 - p0
        area = float(cross2d(e1, e2))
        if abs(area) < 1e-15:
            return
        lo = np.minimum(np.minimum(p0, p1), p2)
        hi = np.maximum(np.maximum(p0, p1), p2)
        i0 = max(int((lo[0] - x0) / cell), 0)
        i1 = min(int((hi[0] - x0) / cell) + 1, nx)
        j0 = max(int((lo[1] - y0) / cell), 0)
        j1 = min(int((hi[1] - y0) / cell) + 1, ny)
        if i0 >= i1 or j0 >= j1:
            return
        cx = x0 + (np.arange(i0, i1) + 0.5) * cell
        cy = y0 + (np.arange(j0, j1) + 0.5) * cell
        gx, gy = np.meshgrid(cx, cy)
        r = np.stack([gx - p0[0], gy - p0[1]], axis=-1)
        w1 = cross2d(r, e2) / area              # cross(r, e2) / cross(e1, e2)
        w2 = cross2d(e1, r) / area              # cross(e1, r) / cross(e1, e2)
        w0 = 1.0 - w1 - w2
        eps = -0.05  # slight overdraw so edge cells are covered
        inside = (w0 >= eps) & (w1 >= eps) & (w2 >= eps)
        z = w0 * z0 + w1 * z1 + w2 * z2
        sub = buf[j0:j1, i0:i1]
        np.maximum(sub, np.where(inside, z, -np.inf), out=sub)

    for i in range(n - 1):
        for j in range(m - 1):
            a, b, c, d = P[i, j], P[i + 1, j], P[i + 1, j + 1], P[i, j + 1]
            za, zb, zc, zd = D[i, j], D[i + 1, j], D[i + 1, j + 1], D[i, j + 1]
            raster_tri(a, b, c, za, zb, zc)
            raster_tri(a, c, d, za, zc, zd)
    return _DepthBuffer(buf, (float(x0), float(y0)), float(cell))


def _quad_depth_spread(D: np.ndarray) -> float:
    """Max corner-depth range over quads — bounds the z-buffer interpolation
    error for a sample lying on the surface itself."""
    di = np.abs(np.diff(D, axis=0))
    dj = np.abs(np.diff(D, axis=1))
    return float(max(di.max(initial=0.0), dj.max(initial=0.0),
                     np.abs(D[1:, 1:] - D[:-1, :-1]).max(initial=0.0)))


def wireframe_items(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, cam: Camera,
                    stride: tuple[int, int] = (1, 1), hidden: str | None = None,
                    grid: int = 256, tol: float | None = None,
                    depth_bias: float = 0.04, **curve_kw) -> list[tuple[float, Curve]]:
    """u- and v-isolines with hidden-line removal against the surface itself.

    A segment is hidden iff its midpoint depth < z-buffer depth at its screen
    cell - tol (tol defaults to the max per-quad depth spread, so a sample on
    the surface never occludes itself). hidden=None drops hidden segments;
    a dash spec (e.g. "dashed") keeps them dashed and muted.
    """
    db = _depth_buffer(X, Y, Z, cam, grid)
    n, m = Z.shape
    pts3 = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    pts2, depth = project(pts3, cam)
    P = pts2.reshape(n, m, 2)
    D = depth.reshape(n, m)
    if tol is None:
        tol = 1.001 * _quad_depth_spread(D) + 1e-9

    def lines(idx: range, count: int):
        out = sorted(set(idx) | {count - 1})
        return out

    items: list[tuple[float, Curve]] = []

    def emit(p2_line: np.ndarray, d_line: np.ndarray) -> None:
        mid2 = 0.5 * (p2_line[:-1] + p2_line[1:])
        midd = 0.5 * (d_line[:-1] + d_line[1:])
        occluded = midd < db.lookup(mid2) - tol
        for k in range(len(mid2)):
            seg = p2_line[k:k + 2]
            if not occluded[k]:
                items.append((float(midd[k]) + depth_bias, Curve(seg, **curve_kw)))
            elif hidden is not None:
                _, o_mult, w_mult = DEFAULT_STYLE.hidden_variant(
                    curve_kw.get("role", Role.CONTENT))
                kw = dict(curve_kw)
                kw["dash"] = hidden
                kw["opacity"] = kw.get("opacity", 1.0) * o_mult
                kw["width_scale"] = kw.get("width_scale", 1.0) * w_mult
                items.append((float(midd[k]), Curve(seg, **kw)))

    for i in lines(range(0, n, stride[0]), n):
        emit(P[i, :], D[i, :])
    for j in lines(range(0, m, stride[1]), m):
        emit(P[:, j], D[:, j])
    return items


def label3(latex: str, anchor3, cam: Camera, **label_kw) -> MathLabel:
    """A MathLabel anchored at a world-space point.

    Replaces the project-unpack-relabel idiom: the anchor is projected
    here, and everything else (ha/va/offset_px/role/size_pt) passes
    through to MathLabel unchanged. offset_px stays canvas px, +y down.
    """
    (p2,), _ = project(np.asarray([anchor3], dtype=float), cam)
    return MathLabel(latex, (float(p2[0]), float(p2[1])), **label_kw)


def vector3(tail3, tip3, cam: Camera, **vector_kw) -> Vector:
    """A Vector with world-space endpoints, projected at build time."""
    (t2, p2), _ = project(np.asarray([tail3, tip3], dtype=float), cam)
    return Vector((float(t2[0]), float(t2[1])), (float(p2[0]), float(p2[1])),
                  **vector_kw)


def as_floor(group: list[tuple[float, object]]) -> list[tuple[float, object]]:
    """Force a depth-tagged group behind everything in compose() — the
    ground-plane idiom, without magic depth constants in figure programs."""
    return [(-math.inf, it) for _, it in group]


def drop_shadow(pts3: np.ndarray, cam: Camera, z0: float = 0.0,
                opacity: float = 0.16, color: str | None = None,
                role: Role = Role.MUTED,
                depth_bias: float | None = None) -> list[tuple[float, FilledCurve]]:
    """Contact shadow: the convex hull of the geometry's vertical footprint
    on the plane z=z0, projected and depth-tagged strictly behind every
    point of the geometry — the cue that places a floating object ON its
    floor (compose() after as_floor ground, before the object).

    depth_bias=None keeps that strictly-behind tag. A positive value tags
    the shadow at its own mean depth + bias instead — for a shadow cast
    onto another object's TOP FACE (same plane, so same mean depth): the
    bias paints it over its carrier face, while the caster above remains
    strictly nearer.
    """
    from scipy.spatial import ConvexHull

    pts3 = np.asarray(pts3, dtype=float)
    foot = pts3[ConvexHull(pts3[:, :2]).vertices, :2]
    shadow3 = np.column_stack([foot, np.full(len(foot), z0)])
    pts2, depth = project(shadow3, cam)
    if depth_bias is not None:
        tag = float(depth.mean()) + depth_bias
    else:
        # min footprint depth <= min object depth (same xy, z above the
        # floor for elev > 0), so this tag paints before every object quad
        tag = float(depth.min()) - 1e-3 * (float(np.ptp(depth)) + 1e-9)
    return [(tag, FilledCurve(pts2, role=role, opacity=opacity,
                              outline=False, color=color))]
