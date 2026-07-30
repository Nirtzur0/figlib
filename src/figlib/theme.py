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
WASH = "wash"                       # large-area region fill, paper-blended


def _lerp_hex(stops: list[str], t: float) -> str:
    """Walk a stop list in OKLab, not in hex channels.

    A ramp's semantic contract is ORDER, and the coordinate carrying order is
    OKLab L — so that is the space the interpolation has to happen in.
    Interpolating '#rrggbb' channels handed the middle of every ramp to the
    sRGB transfer curve: the RISO level-curve ramp's sampled lightness
    wandered 0.010 off the straight perceptual path between its stops and
    reversed by ~0.001 inside the mustard->brick leg. Mixing in OKLab makes
    L(t) exactly piecewise linear, so equal steps in t are equal steps in
    perceived lightness. Endpoints and stop colors are unchanged — the
    palettes still look like themselves, only the middles are smoother.
    """
    from .color import mix_oklab

    t = min(max(t, 0.0), 1.0)
    x = t * (len(stops) - 1)
    j = min(int(x), len(stops) - 2)
    return mix_oklab(stops[j], stops[j + 1], x - j)


# The cyclic phase channel maps modulus onto lightness through a sawtooth in
# log2|f|, so one period is one OCTAVE — a doubling of the modulus. A decade
# was the alternative; octaves win because the interesting structure near a
# zero or a pole is a *ratio* structure and doublings put roughly three times
# as many contours in the same dynamic range, which is what makes the rings
# around z = 1 legible rather than a single wash.
PHASE_PERIOD_OCTAVES = 1.0
# |f| outside 2^±40 is a zero or a pole as far as any raster is concerned;
# clamping there keeps log2 finite so those pixels get a color, not a hole.
_PHASE_LOG_CLAMP = 40.0


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

    # --- cyclic phase channel (domain coloring) ---------------------------
    # A *periodic* quantity cannot ride the ordered ramp: arg = -pi and
    # arg = +pi are the same direction and must get the same color, which no
    # monotone scale can do. So phase gets its own channel — the OKLCh hue
    # circle at constant L and C, which is the one hue path that is
    # perceptually even all the way round (HSV's is not: its yellow is far
    # lighter than its blue, so an HSV phase portrait fakes structure at
    # arg ~ pi/2 that the function does not have).
    # C is held constant around the circle rather than pushed to each hue's
    # maximum, because a chroma that varies with phase is a second signal in
    # a channel meant to carry none. sRGB is the binding constraint: at
    # constant L the largest chroma available at EVERY hue is only ~0.106 at
    # L = 0.60, rising to ~0.130 at L = 0.74 — so each theme's C is set just
    # under the floor of its own sawtooth band and the two themes cannot
    # differ much here. They differ in hue origin, base lightness and band
    # depth instead.
    phase_lightness: float = 0.70     # L at the middle of the sawtooth band
    phase_chroma: float = 0.095       # C held around the whole circle
    phase_hue_deg: float = 250.0      # hue at arg = 0 — the theme's signature
    phase_band: float = 0.085         # peak-to-peak L of the per-octave sawtooth
    phase_trend: float = 0.16         # L amplitude of the zero-dark/pole-light trend
    # tanh scale of that trend, in octaves. Three, not the ten a "cover the
    # dynamic range" instinct suggests: the trend has to have visibly picked
    # a side by the time the eye is a fifth of the frame from a singularity,
    # and |f| has already moved 3-4 octaves by then. Set wide, the polarity
    # only resolves in the last few pixels, where the sawtooth is aliasing
    # anyway, and zeros and poles read alike.
    phase_trend_octaves: float = 3.0

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

    def wash(self, i: int | None = None, strength: float = 0.38) -> Hue:
        """Large-area region fill: a hue pre-blended toward the paper,
        returned OPAQUE — basin shading, region identity, ground washes.

        This exists because the alternative — role ink under a 0.18 alpha
        tint — goes muddy on textured paper, so figures avoid regions
        entirely and the area channel sits unused. Blending in the theme
        (against the light paper stop) keeps the result printable and lets
        the contrast gate measure it directly: a vanished wash still fails
        the perceptible floor.

        i indexes the same fixed slots as categorical(), so a wash can
        carry correspondence with a stroke hue (basin i matches sink i);
        None gives a neutral content-ink wash. strength is the ink
        fraction of the blend; the default is the smallest value at which
        EVERY slot clears the 1.3:1 perceptible floor on the worst paper
        stop of both house themes (light slots like RISO's sage bind at
        0.38; darker slots clear from ~0.2). Weaker washes are legal but
        slot-dependent — the color gate is the arbiter, not the eye.
        """
        if i is None:
            base = self.ink(Role.CONTENT).color
        else:
            if not 0 <= i < len(self.categorical_stops):
                raise IndexError(
                    f"wash slot {i} outside the {len(self.categorical_stops)} "
                    f"fixed slots — washes carry correspondence, same rules "
                    f"as categorical()")
            base = self.categorical_stops[i]
        paper = self.paper[0] if self.paper else self.background
        if not paper.startswith("#"):
            paper = "#ffffff"
        return Hue(_lerp_hex([paper, base], strength), WASH)

    def phase(self, arg, mag):
        """(arg, mag) -> sRGB in [0, 1], shape (..., 3). Vectorized.

        The cyclic counterpart of `ramp`. `arg` is a phase in [-pi, pi] and
        `mag` a modulus > 0; together they are one complex number per pixel,
        so this runs per-pixel and returns an RGB array rather than a `Hue`.

        Mechanism, in three terms:

        - **hue = arg**, rigidly. `h = phase_hue_deg + arg` on the OKLCh
          circle at fixed L and C, so equal phase differences are equal
          perceived hue differences and the wrap at ±pi is seamless by
          construction. Which hue sits at arg = 0 is the theme's choice, and
          is what makes a CLEAN portrait and a RISO portrait recognizably
          the same figure in two houses.
        - **lightness = a sawtooth in log2|f|**, one period per octave
          (`PHASE_PERIOD_OCTAVES`). L rises across each doubling and steps
          back at |f| = 2^k, so the level sets |f| = 2^k draw themselves as
          sharp contours. Modulus information for free: no extra ink, no
          second pass, and the contours crowd exactly where |f| moves fast.
        - **lightness += a saturating trend in log2|f|**, `tanh(u / K)`.
          The sawtooth alone is periodic, so a zero and a pole look
          identical up close — both are a pile of converging rings. The
          trend breaks that symmetry: zeros go dark, poles go light, and the
          two are told apart at a glance. tanh rather than a linear term so
          the range stays bounded no matter how wild |f| gets.

        Chroma carries nothing, which is why gamut handling reduces C
        (hue and L held exactly) rather than clipping channels. Near a zero
        or a pole the trend pushes L past what sRGB can hold at full chroma,
        so those cores desaturate toward black and white — which is the
        classic domain-coloring look and is here a consequence, not a rule.
        """
        import numpy as np

        from .color import oklch_to_rgb

        arg = np.nan_to_num(np.asarray(arg, dtype=float), nan=0.0)
        mag = np.asarray(mag, dtype=float)
        finite = np.isfinite(mag)
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.log2(np.where(finite & (mag > 0.0), mag, 1.0))
        u = np.where(finite, u, _PHASE_LOG_CLAMP)          # poles
        u = np.where(finite & (mag <= 0.0), -_PHASE_LOG_CLAMP, u)   # zeros
        u = np.clip(u, -_PHASE_LOG_CLAMP, _PHASE_LOG_CLAMP)

        saw = np.mod(u / PHASE_PERIOD_OCTAVES, 1.0)
        L = (self.phase_lightness
             + self.phase_band * (saw - 0.5)
             + self.phase_trend * np.tanh(u / self.phase_trend_octaves))
        h = np.radians(self.phase_hue_deg) + arg
        return oklch_to_rgb(L, self.phase_chroma, h)

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
                background=style.background, transparent=style.transparent, **kw)


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
    # Phase circle, riso identity: mustard at arg = 0 (the house ochre sits
    # at OKLCh hue ~80 deg), a lower base lightness and a heavier sawtooth
    # than CLEAN — a riso print's whole look is banding, so the modulus
    # contours are wanted loud here and quiet there. The deeper band reaches
    # L = 0.60, where sRGB caps every-hue chroma at 0.106, which is what sets
    # C = 0.104 (still above CLEAN's 0.095: the portrait reads as ink).
    phase_lightness=0.66,
    phase_chroma=0.104,
    phase_hue_deg=80.0,
    phase_band=0.115,
    phase_trend=0.17,
)


def transparent_variant(theme: Theme) -> Theme:
    """The same ink and palettes, no paper: ink on alpha, for a figure whose
    embedding document owns the background. Grain still rides — it is ink."""
    from dataclasses import replace
    return replace(theme, transparent=True)


def opaque_variant(theme: Theme) -> Theme:
    """The theme's own ground: paper gradient, grain, casings and halos.
    This is the default; kept as an explicit spelling."""
    from dataclasses import replace
    return replace(theme, transparent=False)


RISO_PAPER = opaque_variant(RISO)      # == RISO; the ground is now the default
CLEAN_PAPER = opaque_variant(CLEAN)
RISO_CLEAR = transparent_variant(RISO)
CLEAN_CLEAR = transparent_variant(CLEAN)

# Back-compat: transparency is now the default, so these are the base themes.
RISO_T = RISO
CLEAN_T = CLEAN


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
