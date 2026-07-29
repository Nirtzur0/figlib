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


def _stroke_outside_symbols(el: ET.Element, color: str, width: float) -> None:
    """Stroke the drawable elements of a ziamath subtree, EXCEPT inside
    <symbol> definitions — symbols are shared by id across glyph copies, so
    mutating them would leak the stroke onto every copy. Stroke set on the
    referencing <use> inherits into the symbol's shadow tree instead."""
    for ch in el:
        tag = ch.tag.split("}")[-1]
        if tag == "symbol":
            continue
        if tag in ("use", "path", "polygon", "line", "rect"):
            ch.set("stroke", color)
            ch.set("stroke-width", f"{width:.2f}")
            ch.set("stroke-linejoin", "round")
        _stroke_outside_symbols(ch, color, width)


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
    halo_color: str | None = None,
    halo_width: float = 0.0,
) -> None:
    """Draw a math label onto an ElementTree SVG root at canvas px coords.

    With halo_color set, a copy of the glyphs is drawn first, filled AND
    stroked in that color (stroke extends halo_width px beyond the glyph
    outline) — the cartographic casing that keeps labels legible on ink."""
    if halo_color is not None and halo_width > 0.0:
        n_before = len(svg_root)
        h = _latex(latex, size_pt, halo_color)
        h.drawon(svg_root, x_px, y_px, halign=halign, valign=valign)
        for added in list(svg_root)[n_before:]:
            _stroke_outside_symbols(added, halo_color, 2.0 * halo_width)
    m = _latex(latex, size_pt, color)
    m.drawon(svg_root, x_px, y_px, halign=halign, valign=valign)
