"""Orthographic 3D: projection, painter's algorithm, Lambertian shading.

Produces ordinary 2D scene items (FilledCurve quads, Curve segments)
sorted far-to-near, so the whole 2D pipeline — gates included — applies
unchanged to 3D figures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .scene import Curve, FilledCurve
from .style import Role


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
