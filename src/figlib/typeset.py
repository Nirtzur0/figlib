"""Math typesetting via ziamath: LaTeX -> SVG with exact metrics.

The exact glyph bounding boxes are what make the mechanical gate's
collision detection deterministic rather than heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from xml.etree import ElementTree as ET

import ziamath

PX_PER_PT = 96.0 / 72.0


@dataclass(frozen=True)
class TypesetLabel:
    latex: str
    size_pt: float
    width_pt: float
    height_pt: float

    @property
    def width_px(self) -> float:
        return self.width_pt * PX_PER_PT

    @property
    def height_px(self) -> float:
        return self.height_pt * PX_PER_PT


@lru_cache(maxsize=512)
def _latex(latex: str, size_pt: float, color: str) -> ziamath.Latex:
    return ziamath.Latex(latex, size=size_pt * PX_PER_PT, color=color)


def render_math(latex: str, size_pt: float = 11.0, color: str = "#1a1a1a") -> TypesetLabel:
    m = _latex(latex, size_pt, color)
    w, h = m.getsize()  # in svg units == px at our chosen size
    return TypesetLabel(latex, size_pt, width_pt=w / PX_PER_PT, height_pt=h / PX_PER_PT)


def draw_math(
    svg_root: ET.Element,
    latex: str,
    x_px: float,
    y_px: float,
    size_pt: float,
    color: str,
    halign: str = "left",
    valign: str = "base",
) -> None:
    """Draw a math label onto an ElementTree SVG root at canvas px coords."""
    m = _latex(latex, size_pt, color)
    m.drawon(svg_root, x_px, y_px, halign=halign, valign=valign)
