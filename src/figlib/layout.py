"""Math -> canvas mapping. Equal aspect, y-flip, margins from content."""

from __future__ import annotations

import numpy as np

from .scene import AngleMark, Curve, MathLabel, Point, RightAngleMark, Scene, Vector


def geometry_extents(scene: Scene) -> tuple[tuple[float, float], tuple[float, float]]:
    """Extents of the drawable geometry in math coords (labels excluded;
    label overflow is caught by the mechanical gate)."""
    xs: list[float] = []
    ys: list[float] = []
    for it in scene.items:
        if isinstance(it, Curve):
            xs.extend(it.pts[:, 0])
            ys.extend(it.pts[:, 1])
        elif isinstance(it, Vector):
            xs.extend([it.tail[0], it.tip[0]])
            ys.extend([it.tail[1], it.tip[1]])
        elif isinstance(it, Point):
            xs.append(it.xy[0])
            ys.append(it.xy[1])
        elif isinstance(it, (RightAngleMark, AngleMark)):
            cx, cy = (it.corner if isinstance(it, RightAngleMark) else it.center)
            xs.append(cx)
            ys.append(cy)
        elif isinstance(it, MathLabel):
            xs.append(it.anchor[0])
            ys.append(it.anchor[1])
    if scene.xlim is not None:
        xlim = scene.xlim
    else:
        xlim = (float(min(xs)), float(max(xs)))
    if scene.ylim is not None:
        ylim = scene.ylim
    else:
        ylim = (float(min(ys)), float(max(ys)))
    return xlim, ylim


class Transform:
    """Affine math->canvas map: equal aspect, y flipped, uniform padding."""

    def __init__(self, scene: Scene, width_px: float = 900, pad_frac: float = 0.08):
        (x0, x1), (y0, y1) = geometry_extents(scene)
        dx = max(x1 - x0, 1e-12)
        dy = max(y1 - y0, 1e-12)
        pad_x = pad_frac * dx
        pad_y = pad_frac * dy
        x0, x1, y0, y1 = x0 - pad_x, x1 + pad_x, y0 - pad_y, y1 + pad_y
        self.scale = width_px / (x1 - x0)
        self.canvas_w = width_px
        self.canvas_h = (y1 - y0) * self.scale
        self._x0 = x0
        self._y1 = y1

    def to_canvas(self, xy: tuple[float, float]) -> tuple[float, float]:
        return (
            (xy[0] - self._x0) * self.scale,
            (self._y1 - xy[1]) * self.scale,
        )

    def to_canvas_arr(self, pts: np.ndarray) -> np.ndarray:
        out = np.empty_like(pts, dtype=float)
        out[:, 0] = (pts[:, 0] - self._x0) * self.scale
        out[:, 1] = (self._y1 - pts[:, 1]) * self.scale
        return out
