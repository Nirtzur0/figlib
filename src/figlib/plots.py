"""Data plots ON scene primitives — no second rendering path.

An axis is Curves + MathLabels; a series is a Curve, FilledCurve, or
Points (architecture.md). Axes owns the data->scene transform: log
scales happen HERE, so scene coordinates stay affine and layout's
Transform stays dumb. Set Scene.height_px for a free aspect ratio and
Scene x/ylim from scene_xlim/scene_ylim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .scene import Curve, FilledCurve, Item, MathLabel, Point
from .style import Role


def nice_ticks(lo: float, hi: float, target: int = 5) -> list[float]:
    """Tick locations at a 1/2/5 x 10^k step covering [lo, hi]."""
    span = hi - lo
    if span <= 0:
        return [lo]
    raw = span / max(target, 1)
    mag = 10.0 ** math.floor(math.log10(raw))
    step = 10.0 * mag
    for m in (1.0, 2.0, 5.0, 10.0):
        if span / (m * mag) <= target + 0.5:
            step = m * mag
            break
    first = math.ceil(lo / step - 1e-9) * step
    ticks = np.arange(first, hi + 1e-9 * step, step)
    return [float(t) for t in ticks]


def _fmt_num(v: float, step: float) -> str:
    if abs(v) < 1e-12:
        return "0"
    decimals = max(0, -math.floor(math.log10(step) + 1e-9))
    return f"{v:.{decimals}f}"


def _decades(lo: float, hi: float) -> list[float]:
    k0 = math.ceil(math.log10(lo) - 1e-9)
    k1 = math.floor(math.log10(hi) + 1e-9)
    return [10.0 ** k for k in range(k0, k1 + 1)]


@dataclass(frozen=True)
class Axes:
    """Data->scene mapping plus emitters for axis furniture and series.

    Everything returned is an ordinary scene item; the figure program
    decides placement, roles, and hues exactly as for geometry."""
    xlim: tuple[float, float]          # data coords
    ylim: tuple[float, float]
    xscale: str = "linear"             # "linear" | "log"
    yscale: str = "linear"

    def tx(self, x):
        x = np.asarray(x, dtype=float)
        return np.log10(x) if self.xscale == "log" else x

    def ty(self, y):
        y = np.asarray(y, dtype=float)
        return np.log10(y) if self.yscale == "log" else y

    def to_scene(self, x, y) -> np.ndarray:
        return np.column_stack([self.tx(x), self.ty(y)])

    @property
    def scene_xlim(self) -> tuple[float, float]:
        return (float(self.tx(self.xlim[0])), float(self.tx(self.xlim[1])))

    @property
    def scene_ylim(self) -> tuple[float, float]:
        return (float(self.ty(self.ylim[0])), float(self.ty(self.ylim[1])))

    # -- tick locators (data coords) ---------------------------------------

    def xticks(self, target: int = 5) -> list[float]:
        if self.xscale == "log":
            return _decades(*self.xlim)
        return nice_ticks(*self.xlim, target)

    def yticks(self, target: int = 4) -> list[float]:
        if self.yscale == "log":
            return _decades(*self.ylim)
        return nice_ticks(*self.ylim, target)

    # -- axis furniture ----------------------------------------------------

    def _tick_label(self, v: float, ticks: list[float], scale: str) -> str:
        if scale == "log":
            return f"10^{{{round(math.log10(v))}}}"
        step = ticks[1] - ticks[0] if len(ticks) > 1 else (v or 1.0)
        return _fmt_num(v, step)

    def frame(self, tick_frac: float = 0.022, grid: bool = False,
              label_gap_px: float = 4.0, target: tuple[int, int] = (5, 4)) -> list[Item]:
        """Spines, ticks, tick labels — and gridlines if asked. Annotation
        ink for the working parts, FRAME ink for the grid."""
        (sx0, sx1), (sy0, sy1) = self.scene_xlim, self.scene_ylim
        tlen_y = tick_frac * (sy1 - sy0)       # x-axis ticks extend in y
        tlen_x = tick_frac * (sx1 - sx0)       # y-axis ticks extend in x
        items: list[Item] = [
            Curve(np.array([[sx0, sy0], [sx1, sy0]]), role=Role.ANNOTATION),
            Curve(np.array([[sx0, sy0], [sx0, sy1]]), role=Role.ANNOTATION),
        ]
        xt, yt = self.xticks(target[0]), self.yticks(target[1])
        for v in xt:
            x = float(self.tx(v))
            items.append(Curve(np.array([[x, sy0], [x, sy0 - tlen_y]]),
                               role=Role.ANNOTATION))
            if grid:
                items.append(Curve(np.array([[x, sy0], [x, sy1]]), role=Role.FRAME))
            items.append(MathLabel(self._tick_label(v, xt, self.xscale),
                                   (x, sy0 - tlen_y), role=Role.ANNOTATION,
                                   ha="center", va="top",
                                   offset_px=(0.0, label_gap_px)))
        for v in yt:
            y = float(self.ty(v))
            items.append(Curve(np.array([[sx0, y], [sx0 - tlen_x, y]]),
                               role=Role.ANNOTATION))
            if grid:
                items.append(Curve(np.array([[sx0, y], [sx1, y]]), role=Role.FRAME))
            items.append(MathLabel(self._tick_label(v, yt, self.yscale),
                                   (sx0 - tlen_x, y), role=Role.ANNOTATION,
                                   ha="right", va="center",
                                   offset_px=(-label_gap_px, 0.0)))
        return items

    def xlabel(self, latex: str, gap_px: float = 26.0, **label_kw) -> MathLabel:
        (sx0, sx1), (sy0, _) = self.scene_xlim, self.scene_ylim
        label_kw.setdefault("role", Role.ANNOTATION)
        return MathLabel(latex, ((sx0 + sx1) / 2.0, sy0), ha="center", va="top",
                         offset_px=(0.0, gap_px), **label_kw)

    def ylabel(self, latex: str, gap_px: float = 38.0, **label_kw) -> MathLabel:
        (sx0, _), (sy0, sy1) = self.scene_xlim, self.scene_ylim
        label_kw.setdefault("role", Role.ANNOTATION)
        return MathLabel(latex, (sx0, (sy0 + sy1) / 2.0), ha="center", va="bottom",
                         offset_px=(-gap_px, 0.0), angle_deg=90.0, **label_kw)

    # -- series ------------------------------------------------------------

    def line(self, x, y, **curve_kw) -> Curve:
        return Curve(self.to_scene(x, y), **curve_kw)

    def band(self, x, y_lo, y_hi, **fill_kw) -> FilledCurve:
        """Error band / ensemble envelope: upper edge forward, lower edge
        back, as one closed FilledCurve."""
        fill_kw.setdefault("outline", False)
        loop = np.vstack([self.to_scene(x, y_hi), self.to_scene(x, y_lo)[::-1]])
        return FilledCurve(loop, **fill_kw)

    def scatter(self, x, y, **point_kw) -> list[Point]:
        return [Point((float(px), float(py)), **point_kw)
                for px, py in self.to_scene(x, y)]

    def hist(self, values, bins: int = 20, density: bool = True,
             **fill_kw) -> FilledCurve:
        """Histogram as one step-outline polygon, closed down to y=0."""
        h, edges = np.histogram(np.asarray(values, dtype=float), bins=bins,
                                density=density)
        xs = np.repeat(edges, 2)
        ys = np.concatenate([[0.0], np.repeat(h, 2), [0.0]])
        fill_kw.setdefault("outline", False)
        return FilledCurve(self.to_scene(xs, ys), **fill_kw)
