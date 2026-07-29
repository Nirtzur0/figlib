"""Polyhedral solids: extruded prisms, boxes, cylinders as shaded faces.

Each visible face becomes ONE FilledCurve whose fill is a linear Gradient
aligned with the light direction projected into the face plane. A flat
face under directional light has CONSTANT Lambert shade — the within-face
drift is stylistic, not physical. So the face's true Lambert value sets
its base tone and the gradient drifts +-grad_amp around it: physics sets
the tone, the gradient adds the glow.

Output is depth-tagged (depth, FilledCurve) pairs — the same contract as
surface3d.surface_items, so compose/drop_shadow/as_floor apply unchanged.
"""

from __future__ import annotations

import numpy as np

from .scene import FilledCurve, Gradient
from .shading import Ramp, quantize
from .style import Role
from .surface3d import LIGHT_DIR, Camera, project


def _lambert(n_hat: np.ndarray) -> float:
    # half-Lambert, not surface3d's max(0, n.L): a clipped Lambert floors
    # every back-lit face at the same ambient tone, so two shadow faces of
    # a box (or half a cylinder's facets) collapse to one color and the
    # form reads flat. 0.5(1 + n.L) keeps every orientation distinct.
    return 0.5 * (1.0 + float(n_hat @ LIGHT_DIR))


def face_item(poly3: np.ndarray, cam: Camera, ramp: Ramp, *,
              grad_amp: float = 0.12, grain: float = 0.0,
              edge: str | None = None, edge_width: float = 0.35,
              stops: int = 5) -> tuple[float, FilledCurve] | None:
    """One planar face -> (mean depth, FilledCurve); None when backfacing.

    Winding defines the outward normal (right-hand rule); faces whose
    normal points away from the camera are culled, which is the whole
    hidden-surface story for a convex solid.
    """
    poly3 = np.asarray(poly3, dtype=float)
    n = np.cross(poly3[1] - poly3[0], poly3[2] - poly3[0])
    nn = float(np.linalg.norm(n))
    if nn == 0.0:
        return None
    n_hat = n / nn
    _, _, toward = cam.axes()
    if float(n_hat @ toward) <= 1e-9:
        return None
    t = _lambert(n_hat)
    pts2, depth = project(poly3, cam)

    grad = None
    if grad_amp > 0.0:
        g3 = LIGHT_DIR - float(LIGHT_DIR @ n_hat) * n_hat
        gn = float(np.linalg.norm(g3))
        if gn > 1e-6:
            g_hat = g3 / gn
            c = poly3.mean(axis=0)
            s = float(np.abs((poly3 - c) @ g_hat).max())
            if s > 1e-9:
                axis2, _ = project(np.array([c - s * g_hat, c + s * g_hat]), cam)
                if float(np.hypot(*(axis2[1] - axis2[0]))) > 1e-6:
                    lo = max(t - grad_amp, 0.0)
                    hi = min(t + grad_amp, 1.0)
                    grad = Gradient.from_ramp(ramp, tuple(axis2[0]),
                                              tuple(axis2[1]),
                                              t_range=(lo, hi), n=stops)
    fill = ramp(t)
    # edge=None seals each face with a hairline of its own base color:
    # adjacent facets otherwise leave antialiasing seams (paper-colored
    # pinstripes down a faceted cylinder)
    return (float(depth.mean()), FilledCurve(
        pts2, role=Role.CONTENT, opacity=1.0, outline=False,
        color=fill, gradient=grad, grain=grain,
        edge_color=edge if edge is not None else fill,
        edge_width=edge_width))


def extrude_items(poly2: np.ndarray, z0: float, z1: float, cam: Camera,
                  ramp: Ramp, *, side_grad_amp: float = 0.12,
                  cap_grad_amp: float = 0.12, bands: int | None = None,
                  grain: float = 0.0, edge: str | None = None,
                  edge_width: float = 0.35) -> list[tuple[float, FilledCurve]]:
    """A CCW polygon (viewed from +z) extruded from z0 to z1.

    bands quantizes the SIDE ramp only (the posterized-cylinder look);
    caps keep the smooth ramp. side_grad_amp=0 gives flat facets.
    """
    poly2 = np.asarray(poly2, dtype=float)
    m = len(poly2)
    side_ramp = ramp
    if bands:
        # posterize across the VISIBLE facets' tonal sweep, not [0, 1]:
        # a side mostly in shadow would otherwise fall inside one band
        _, _, toward = cam.axes()
        ts = []
        for i in range(m):
            e = poly2[(i + 1) % m] - poly2[i]
            n = np.array([e[1], -e[0], 0.0])
            nn = float(np.linalg.norm(n))
            if nn > 0 and float((n / nn) @ toward) > 1e-9:
                ts.append(_lambert(n / nn))
        if ts:
            side_ramp = quantize(ramp, bands, span=(min(ts), max(ts)))
    kw = dict(grain=grain, edge=edge, edge_width=edge_width)
    items: list[tuple[float, FilledCurve]] = []
    for i in range(m):
        a, b = poly2[i], poly2[(i + 1) % m]
        quad3 = np.array([[a[0], a[1], z0], [b[0], b[1], z0],
                          [b[0], b[1], z1], [a[0], a[1], z1]])
        it = face_item(quad3, cam, side_ramp, grad_amp=side_grad_amp, **kw)
        if it is not None:
            items.append(it)
    top = np.column_stack([poly2, np.full(m, z1)])
    bot = np.column_stack([poly2, np.full(m, z0)])[::-1]
    for cap in (top, bot):
        it = face_item(cap, cam, ramp, grad_amp=cap_grad_amp, **kw)
        if it is not None:
            items.append(it)
    return items


def box_items(center, size, cam: Camera, ramp: Ramp,
              **kw) -> list[tuple[float, FilledCurve]]:
    """An axis-aligned box: center (x, y, z), size (sx, sy, sz)."""
    (cx, cy, cz), (sx, sy, sz) = center, size
    rect = np.array([[cx - sx / 2, cy - sy / 2], [cx + sx / 2, cy - sy / 2],
                     [cx + sx / 2, cy + sy / 2], [cx - sx / 2, cy + sy / 2]])
    return extrude_items(rect, cz - sz / 2, cz + sz / 2, cam, ramp, **kw)


def cylinder_items(center, radius: float, height: float, cam: Camera,
                   ramp: Ramp, *, facets: int = 48,
                   **kw) -> list[tuple[float, FilledCurve]]:
    """A vertical cylinder approximated by `facets` flat side quads.

    Sides are flat-shaded per facet (side_grad_amp=0): smoothness comes
    from facet count, the posterized look from bands=n.
    """
    cx, cy, cz = center
    th = np.linspace(0.0, 2.0 * np.pi, facets, endpoint=False)
    poly = np.column_stack([cx + radius * np.cos(th),
                            cy + radius * np.sin(th)])
    return extrude_items(poly, cz - height / 2, cz + height / 2, cam, ramp,
                         side_grad_amp=0.0, **kw)
