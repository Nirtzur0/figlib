"""Math -> canvas mapping. Equal aspect, y-flip, margins from content."""

from __future__ import annotations

import numpy as np

from .scene import (AngleMark, Brace, Callout, Curve, FilledCurve, MathLabel,
                    Point, RasterField, RightAngleMark, Scene, Vector)


def geometry_extents(scene: Scene) -> tuple[tuple[float, float], tuple[float, float]]:
    """Extents of the drawable geometry in math coords (labels excluded;
    label overflow is caught by the mechanical gate)."""
    xs: list[float] = []
    ys: list[float] = []
    for it in scene.items:
        if isinstance(it, (Curve, FilledCurve)):
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
        elif isinstance(it, Brace):
            mid, _, n, d = it.frame()
            apex = mid + d * n
            xs.extend([it.p1[0], it.p2[0], float(apex[0])])
            ys.extend([it.p1[1], it.p2[1], float(apex[1])])
        elif isinstance(it, Callout):
            xs.extend([it.anchor[0], it.target[0]])
            ys.extend([it.anchor[1], it.target[1]])
        elif isinstance(it, RasterField):
            xs.extend(it.extent[0:2])
            ys.extend(it.extent[2:4])
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
        self.scale_x = width_px / (x1 - x0)
        if scene.height_px is None:
            self.scale_y = self.scale_x          # equal aspect
        else:
            self.scale_y = scene.height_px / (y1 - y0)
        self.scale = self.scale_x
        self.canvas_w = width_px
        self.canvas_h = (y1 - y0) * self.scale_y
        self._x0 = x0
        self._y1 = y1

    def to_canvas(self, xy: tuple[float, float]) -> tuple[float, float]:
        return (
            (xy[0] - self._x0) * self.scale_x,
            (self._y1 - xy[1]) * self.scale_y,
        )

    def from_canvas(self, xy: tuple[float, float]) -> tuple[float, float]:
        """Inverse of to_canvas — for diagnostics that must speak math coords."""
        return (xy[0] / self.scale_x + self._x0,
                self._y1 - xy[1] / self.scale_y)

    def to_canvas_arr(self, pts: np.ndarray) -> np.ndarray:
        out = np.empty_like(pts, dtype=float)
        out[:, 0] = (pts[:, 0] - self._x0) * self.scale_x
        out[:, 1] = (self._y1 - pts[:, 1]) * self.scale_y
        return out
