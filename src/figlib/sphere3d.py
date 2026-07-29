"""Sphere projection: depth-tagged 2D items for surface3d.compose().

Orthographic visibility on a sphere is exact: a surface point p is
visible iff (p - center)·toward > 0, with toward from Camera.axes().
Every producer here returns list[(depth, Item)] so spheres, draped
curves, occluded rays, and plane quads all merge through the one
painter's algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scene import Curve, FilledCurve
from .style import DEFAULT_STYLE, Role
from .surface3d import Camera, project


@dataclass(frozen=True)
class Sphere:
    center: np.ndarray
    radius: float


def point_on(sphere: Sphere, direction: np.ndarray) -> np.ndarray:
    """The surface point in the given (unnormalized) direction from center."""
    d = np.asarray(direction, dtype=float)
    return sphere.center + sphere.radius * d / np.linalg.norm(d)


def anchor(pt3: np.ndarray, cam: Camera) -> tuple[float, float]:
    """A 3D point as a 2D scene anchor (for Points / MathLabels)."""
    pts2, _ = project(np.asarray(pt3, dtype=float)[None, :], cam)
    return float(pts2[0, 0]), float(pts2[0, 1])


def circle_points(sphere: Sphere, normal: np.ndarray, offset: float,
                  n: int = 256) -> np.ndarray:
    """Sample the circle at signed distance offset from center along normal.

    offset=0 is a great circle; |offset| must be < radius.
    """
    nh = np.asarray(normal, dtype=float)
    nh = nh / np.linalg.norm(nh)
    assert abs(offset) < sphere.radius, "circle offset must lie inside the sphere"
    rc = np.sqrt(sphere.radius**2 - offset**2)
    seed = np.array([0.0, 0.0, 1.0]) if abs(nh[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(nh, seed)
    u /= np.linalg.norm(u)
    v = np.cross(nh, u)
    th = np.linspace(0.0, 2.0 * np.pi, n)
    return (sphere.center + offset * nh
            + rc * (np.cos(th)[:, None] * u + np.sin(th)[:, None] * v))


def _split_signed(pts3: np.ndarray, signed: np.ndarray,
                  eps: float = 0.0) -> list[tuple[bool, np.ndarray]]:
    """Maximal runs of one visibility state; crossings interpolated on signed.

    signed is linear in position for the on-sphere dot test, so
    interpolated crossing points satisfy signed = 0 to float precision.
    """
    vis = signed >= eps  # a sample exactly on the limb counts visible
    runs: list[tuple[bool, np.ndarray]] = []
    cur: list[np.ndarray] = [pts3[0]]
    cur_vis = bool(vis[0])
    for k in range(1, len(pts3)):
        if bool(vis[k]) == cur_vis:
            cur.append(pts3[k])
            continue
        s0, s1 = float(signed[k - 1]), float(signed[k])
        t = s0 / (s0 - s1) if s0 != s1 else 0.5
        x = pts3[k - 1] + t * (pts3[k] - pts3[k - 1])
        cur.append(x)
        runs.append((cur_vis, np.array(cur)))
        cur = [x, pts3[k]]
        cur_vis = bool(vis[k])
    runs.append((cur_vis, np.array(cur)))
    return [(v, r) for v, r in runs
            if len(r) > 1 and float(np.linalg.norm(np.diff(r, axis=0), axis=1).sum()) > 1e-12]


def visibility_runs(pts3: np.ndarray, sphere: Sphere, cam: Camera,
                    tol: float = 0.0) -> list[tuple[bool, np.ndarray]]:
    """Split an on-sphere polyline into (visible, pts3) runs by the dot test."""
    _, _, toward = cam.axes()
    signed = (np.asarray(pts3, dtype=float) - sphere.center) @ toward
    return _split_signed(np.asarray(pts3, dtype=float), signed, eps=tol)


def _emit(runs: list[tuple[bool, np.ndarray]], cam: Camera, hidden: str | None,
          depth_bias: float, curve_kw: dict) -> list[tuple[float, Curve]]:
    """Runs -> depth-tagged curves; visible biased toward the viewer so they
    beat a filled sphere disc, hidden restyled via Style.hidden_variant
    (dash from the `hidden` arg, which overrides the variant's) or dropped."""
    items: list[tuple[float, Curve]] = []
    _, o_mult, w_mult = DEFAULT_STYLE.hidden_variant(
        curve_kw.get("role", Role.CONTENT))
    for visible, run in runs:
        pts2, depth = project(run, cam)
        if visible:
            items.append((float(depth.mean()) + depth_bias, Curve(pts2, **curve_kw)))
        elif hidden is not None:
            kw = dict(curve_kw)
            kw["dash"] = hidden
            kw["opacity"] = kw.get("opacity", 1.0) * o_mult
            kw["width_scale"] = kw.get("width_scale", 1.0) * w_mult
            items.append((float(depth.mean()), Curve(pts2, **kw)))
    return items


def drape(pts3: np.ndarray, sphere: Sphere, cam: Camera, hidden: str | None = "dashed",
          tol: float = 1e-9, depth_bias: float = 0.04,
          signed: np.ndarray | None = None,
          **curve_kw) -> list[tuple[float, Curve]]:
    """A polyline lying ON the sphere, split into visible/hidden runs.

    `signed` overrides the default camera dot test with a per-sample
    visibility scalar (visible where >= tol), for curves whose hiddenness
    is not just facing-away: e.g. an arc occluded by BOTH the far sheet
    and a cutting plane takes the elementwise min of the two tests. It
    must stay LINEAR in position along the polyline — the split
    interpolates crossings on it, so a nonlinear scalar puts the
    solid/dashed break in the wrong place.
    """
    runs = (_split_signed(np.asarray(pts3, dtype=float),
                          np.asarray(signed, dtype=float), eps=tol)
            if signed is not None
            else visibility_runs(pts3, sphere, cam, tol=tol))
    return _emit(runs, cam, hidden, depth_bias, curve_kw)


def circle_on_sphere(sphere: Sphere, normal: np.ndarray, offset: float, cam: Camera,
                     n: int = 256, hidden: str | None = "dashed",
                     depth_bias: float = 0.04, **curve_kw) -> list[tuple[float, Curve]]:
    """Great (offset=0) or small circle, split at the silhouette."""
    return drape(circle_points(sphere, normal, offset, n=n), sphere, cam,
                 hidden=hidden, depth_bias=depth_bias, **curve_kw)


def occlude(pts3: np.ndarray, sphere: Sphere, cam: Camera,
            hidden: str | None = "dashed", n: int = 256, depth_bias: float = 0.04,
            **curve_kw) -> list[tuple[float, Curve]]:
    """An arbitrary 3D polyline occluded by the sphere.

    A point is hidden iff its view ray meets the sphere nearer than the
    point: screen distance d from the projected center < radius AND
    depth < center_depth + sqrt(radius^2 - d^2) (the front surface).
    """
    pts3 = _resample(np.asarray(pts3, dtype=float), n)
    pts2, depth = project(pts3, cam)
    c2, cd = project(sphere.center[None, :], cam)
    d = np.linalg.norm(pts2 - c2[0], axis=1)
    front = float(cd[0]) + np.sqrt(np.clip(sphere.radius**2 - d**2, 0.0, None))
    # continuous signed visibility: outside the limb by d, else by depth deficit
    signed = np.where(d < sphere.radius,
                      np.maximum(d - sphere.radius, depth - front),
                      d - sphere.radius)
    return _emit(_split_signed(pts3, signed), cam, hidden, depth_bias, curve_kw)


def _resample(pts3: np.ndarray, n: int) -> np.ndarray:
    """Refine a polyline to >= n samples along arc length, keeping every
    original vertex (interpolated points lie exactly on the segments)."""
    if len(pts3) >= n:
        return pts3
    seg = np.linalg.norm(np.diff(pts3, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] == 0.0:
        return pts3
    params = np.union1d(s, np.linspace(0.0, s[-1], n))
    return np.column_stack([np.interp(params, s, pts3[:, k]) for k in range(3)])


def _limb(sphere: Sphere, cam: Camera, n: int) -> tuple[np.ndarray, float]:
    """The silhouette circle in screen coords, and the center depth."""
    c2, cd = project(sphere.center[None, :], cam)
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    pts2 = c2[0] + sphere.radius * np.column_stack([np.cos(th), np.sin(th)])
    return pts2, float(cd[0])


def silhouette(sphere: Sphere, cam: Camera, n: int = 256, depth_bias: float = 0.05,
               **curve_kw) -> list[tuple[float, Curve]]:
    """The limb circle: under orthographic projection an exact circle of the
    sphere's own radius; tagged just nearer than the front so it always draws."""
    pts2, cd = _limb(sphere, cam, n)
    return [(cd + sphere.radius + depth_bias, Curve(pts2, closed=True, **curve_kw))]


def disc(sphere: Sphere, cam: Camera, color: str | None = None, opacity: float = 1.0,
         n: int = 256, role: Role = Role.CONTENT) -> list[tuple[float, FilledCurve]]:
    """A flat-filled limb disc (the sphere's body), tagged at the far side so
    hidden arcs and everything on the sphere paint over it. Flat fill only."""
    pts2, cd = _limb(sphere, cam, n)
    return [(cd - sphere.radius, FilledCurve(pts2, role=role, opacity=opacity,
                                             outline=False, color=color))]
