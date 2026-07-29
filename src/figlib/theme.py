"""Themes: everything about appearance, nothing about content.

A Theme extends Style with paper, palette ramps, and grain. Figures
never hardcode colors — they ask the theme for semantic channels:

    theme.ramp(t)         ordered quantity -> color (level sets, shading)
    theme.categorical(i)  distinct series -> color (correspondence hues)
    theme.surface_shade(t) 3D facet lighting -> color

so the same figure program renders consistently under any theme.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field

from .style import DEFAULT_STYLE, Ink, Role, Style


def _lerp_hex(stops: list[str], t: float) -> str:
    t = min(max(t, 0.0), 1.0)
    x = t * (len(stops) - 1)
    j = min(int(x), len(stops) - 2)
    f = x - j
    a = [int(stops[j][k:k + 2], 16) for k in (1, 3, 5)]
    b = [int(stops[j + 1][k:k + 2], 16) for k in (1, 3, 5)]
    return "#" + "".join(f"{round(u + f * (v - u)):02x}" for u, v in zip(a, b))


@dataclass(frozen=True)
class Theme(Style):
    # paper: vertical gradient behind everything (top, bottom)
    paper: tuple[str, str] = ("#ffffff", "#ffffff")
    # ordered-quantity ramp (level sets, lightness scales)
    ramp_stops: list[str] = field(default_factory=lambda: ["#c6d2e4", "#254a80"])
    # distinct-series hues (correspondence channel)
    categorical_stops: list[str] = field(default_factory=lambda: [
        "#440154", "#31688e", "#1f9e89", "#6ece58", "#dbc932"])
    # 3D facet shading ramp, dark (t=0) to lit (t=1)
    surface_stops: list[str] = field(default_factory=lambda: ["#5f6f8f", "#f4f6fa"])
    surface_edge: str = "#6a6f7a"
    # film grain overlay opacity; 0 disables
    grain: float = 0.0
    # no paper rect, no grain; SVG/PNG keep alpha for embedding in documents
    transparent: bool = False

    def ramp(self, t: float) -> str:
        return _lerp_hex(self.ramp_stops, t)

    def categorical(self, i: int, n: int) -> str:
        return _lerp_hex(self.categorical_stops, 0.5 if n <= 1 else i / (n - 1))

    def surface_shade(self, t: float) -> str:
        return _lerp_hex(self.surface_stops, t)


def _from_style(style: Style, **kw) -> dict:
    return dict(inks=style.inks, label_size_pt=style.label_size_pt,
                point_radius=style.point_radius, arrowhead_len=style.arrowhead_len,
                arrowhead_halfwidth=style.arrowhead_halfwidth,
                background=style.background, **kw)


CLEAN = Theme(**_from_style(DEFAULT_STYLE))

# Risograph: warm paper, sun-bleached saturated hues, printed grain.
RISO = Theme(
    inks={
        Role.CONTENT: Ink("#33302e", 1.7),
        Role.CONSTRUCTION: Ink("#8d8579", 1.1, dash="5 4"),
        Role.ANNOTATION: Ink("#4a443f", 1.1),
        Role.FRAME: Ink("#c4bbae", 0.9, dash="1.5 3.5"),
        Role.ACCENT1: Ink("#3467a3", 1.9),        # sky/indigo
        Role.ACCENT2: Ink("#c8402f", 1.9),        # brick red
        Role.MUTED: Ink("#b3a996", 1.3),
    },
    background="#efe9dc",
    paper=("#f2ede1", "#e9dfcd"),
    ramp_stops=["#e9c46a", "#e76f51", "#9d4260", "#3d405b"],
    categorical_stops=["#3d405b", "#3467a3", "#4d908e", "#90a955",
                       "#e9c46a", "#e76f51", "#c8402f"],
    surface_stops=["#3d3a5c", "#9d4260", "#e76f51", "#e9c46a", "#f4ead0"],
    surface_edge="#7a6f63",
    grain=0.5,
)


def transparent_variant(theme: Theme) -> Theme:
    """The same ink and palettes, no paper — for embedding in other work."""
    from dataclasses import replace
    return replace(theme, transparent=True)


RISO_T = transparent_variant(RISO)
CLEAN_T = transparent_variant(CLEAN)


def grain_tile_datauri(size: int = 140, seed: int = 11) -> str:
    """A tileable monochrome noise PNG as a data URI (the riso grain)."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(seed)
    # sparse dark and light speckle, mostly transparent
    alpha = (rng.random((size, size)) < 0.5).astype("uint8")
    val = rng.integers(0, 2, (size, size), dtype="uint8") * 255
    a = (alpha * rng.integers(10, 26, (size, size))).astype("uint8")
    img = Image.merge("LA", (Image.fromarray(val, "L"), Image.fromarray(a, "L")))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
