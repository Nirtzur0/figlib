"""Scene graph: typed primitives in math coordinates.

Primitives carry semantic roles, never raw stroke settings; only
layout.Transform ever converts to canvas pixels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .style import Role

XY = tuple[float, float]


@dataclass
class Curve:
    pts: np.ndarray                # (N, 2)
    role: Role = Role.CONTENT
    closed: bool = False
    width_scale: float = 1.0
    opacity: float = 1.0


@dataclass
class FilledCurve:
    pts: np.ndarray                # (N, 2), closed implicitly
    role: Role = Role.CONTENT
    opacity: float = 0.18
    outline: bool = True


@dataclass
class Vector:
    tail: XY
    tip: XY
    role: Role = Role.CONTENT
    width_scale: float = 1.0


@dataclass
class Point:
    xy: XY
    role: Role = Role.CONTENT
    filled: bool = True
    radius_scale: float = 1.0


@dataclass
class MathLabel:
    latex: str
    anchor: XY                     # math coords
    role: Role = Role.CONTENT
    size_pt: float | None = None   # None -> style.label_size_pt
    ha: str = "left"               # left | center | right
    va: str = "base"               # top | center | base | bottom
    offset_px: XY = (0.0, 0.0)     # nudge in canvas px, +y down


@dataclass
class RightAngleMark:
    corner: XY
    dir1: XY                       # the two edge directions (math coords)
    dir2: XY
    size: float                    # side length in math units
    role: Role = Role.ANNOTATION


@dataclass
class AngleMark:
    center: XY
    dir1: XY
    dir2: XY
    radius: float                  # math units
    role: Role = Role.ANNOTATION


Item = Curve | FilledCurve | Vector | Point | MathLabel | RightAngleMark | AngleMark


@dataclass
class Scene:
    items: list[Item] = field(default_factory=list)
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    # None -> equal aspect (geometry in the plane); a float -> canvas
    # height in px, axes scaled independently (e.g. (t, x) plots)
    height_px: float | None = None

    def add(self, *items: Item) -> None:
        self.items.extend(items)
