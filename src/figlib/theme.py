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


class Hue(str):
    """A color that remembers which semantic channel produced it.

    grammar.md distinguishes hue-as-correspondence from lightness-as-order,
    but a bare '#rrggbb' in a Scene has lost that distinction — and the two
    are gated differently (correspondence hues must be pairwise separable;
    a ramp only has to be monotone). Tagging at the accessor keeps the
    provenance without any ceremony at the call site: a figure still just
    writes `color=THEME.categorical(0)`, and it is a `str` everywhere else,
    so storage and SVG emission are unchanged.
    """

    channel: str

    def __new__(cls, value: str, channel: str) -> "Hue":
        obj = super().__new__(cls, value)
        obj.channel = channel
        return obj


CORRESPONDENCE = "correspondence"   # hue = identity: same hue, same object
ORDER = "order"                     # lightness = an ordered quantity
SHADE = "shade"                     # 3D facet lighting


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
    # Ordered-quantity ramp (level sets, lightness scales). Unlike a fill
    # ramp, this one is drawn as *line work*, so its light end has to stay
    # above the stroke floor — the old #c6d2e4 sat at 1.5:1 on white, i.e.
    # the low-k level curves were invisible.
    ramp_stops: list[str] = field(default_factory=lambda: ["#7895bf", "#254a80"])
    # Distinct-series hues (correspondence channel), prefix-ordered so the
    # first `correspondence_cap` are pairwise separable. The previous default
    # was the viridis ramp — a *sequential* scale used as categorical slots,
    # which is unsound twice over: its light end sat at 1.7-2.0:1 on white
    # (invisible as line work) and its steps encode order, not identity.
    categorical_stops: list[str] = field(default_factory=lambda: [
        "#3d6fb4",   # blue   (the CLEAN ACCENT1)
        "#b08018",   # ochre
        "#2a8a7f",   # teal
        "#d1495b",   # red    (the CLEAN ACCENT2)
        "#7d4a9e",   # purple
        "#3f8a3f"])  # green
    # 3D facet shading ramp, dark (t=0) to lit (t=1)
    surface_stops: list[str] = field(default_factory=lambda: ["#5f6f8f", "#f4f6fa"])
    surface_edge: str = "#6a6f7a"
    # how many correspondence hues this palette can carry pairwise-distinctly
    correspondence_cap: int = 3
    # film grain overlay opacity; 0 disables
    grain: float = 0.0
    # no paper rect, no grain; SVG/PNG keep alpha for embedding in documents
    transparent: bool = False

    def ramp(self, t: float) -> Hue:
        return Hue(_lerp_hex(self.ramp_stops, t), ORDER)

    def categorical(self, i: int) -> Hue:
        """Slot i of the fixed correspondence order.

        Fixed slots, assigned in order, never interpolated and never
        cycled. Interpolating `i/(n-1)` — the previous behaviour — made a
        series' hue a function of how many series shared the figure, so
        the *same object* took different hues in two panels with different
        counts, breaking the one property this channel exists to carry.
        It also shrank the separation between neighbours as n grew,
        collapsing to ΔE 7.9 by n = 8.

        Past the last slot the answer is never a generated hue (it would
        be indistinguishable from an existing one): fold the tail into one
        'other' hue, facet into panels, or encode identity as hue × dash —
        `Curve.dash` is an independent identity channel for exactly this.
        """
        if not 0 <= i < len(self.categorical_stops):
            raise IndexError(
                f"categorical slot {i} outside the {len(self.categorical_stops)} "
                f"fixed slots; use hue x dash, facet, or fold the tail — "
                f"never a generated hue")
        return Hue(self.categorical_stops[i], CORRESPONDENCE)

    def surface_shade(self, t: float) -> Hue:
        return Hue(_lerp_hex(self.surface_stops, t), SHADE)

    def paper_stops(self) -> list[str]:
        """Every ground a mark could land on — the contrast gate takes the
        worst, since a gradient's dark end is where faint ink disappears."""
        if getattr(self, "transparent", False):
            # Embedded in unknown media; assume the hostile case is white.
            return ["#ffffff"]
        return list(self.paper) if self.paper else [self.background]


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
        Role.CONSTRUCTION: Ink("#7d7568", 1.1, dash="5 4"),
        Role.ANNOTATION: Ink("#4a443f", 1.1),
        Role.FRAME: Ink("#c4bbae", 0.9, dash="1.5 3.5"),
        Role.ACCENT1: Ink("#3467a3", 1.9),        # sky/indigo
        Role.ACCENT2: Ink("#c8402f", 1.9),        # brick red
        Role.MUTED: Ink("#857a63", 1.3),
    },
    background="#efe9dc",
    paper=("#f2ede1", "#e9dfcd"),
    # Level-curve ramp, re-stepped to stay monotone in L *and* above the
    # stroke floor on cream: the old light end (#e9c46a) was 1.27:1, so the
    # low-k level curves were effectively unprinted. The cost is a narrower
    # lightness range (L 0.60->0.38 instead of 0.83->0.38), so hue rotation
    # now carries more of the order signal than lightness alone.
    ramp_stops=["#a17918", "#b73919", "#863852", "#3d405b"],
    # Fixed correspondence slots, prefix-ordered: the first four are
    # pairwise separable (all-pairs ΔE 12.4 under protan/deutan), which is
    # the pairlist that applies here — any two curves in a plane figure can
    # end up side by side, unlike bars in a stack. Slots 5-7 keep the house
    # palette available but collapse against each other under deuteranopia
    # (sage/coral/brick sit at ΔE 1.8-4.7), so past four hues identity must
    # ride a second channel. Each hue was snapped to clear 3:1 on the cream
    # paper by moving lightness and chroma only, holding the hue angle —
    # mustard was at 1.27:1, effectively invisible in print.
    categorical_stops=["#373b67",   # indigo
                       "#a17814",   # ochre  (was #e9c46a, 1.27:1)
                       "#3467a3",   # sky
                       "#388b89",   # teal   (was #4d908e, 2.79:1)
                       "#6f8932",   # sage   (was #90a955, 1.99:1)
                       "#e14a24",   # coral  (was #e76f51, 2.34:1)
                       "#c8402f"],  # brick
    # Hues beyond this many are not pairwise distinguishable in this
    # palette; the gate rejects a figure that uses more.
    correspondence_cap=4,
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
