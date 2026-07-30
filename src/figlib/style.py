"""House style: semantic roles resolved to ink.

The grammar (from Needham): solid = content, dashed = construction,
dotted = frame. Color is semantic or absent — the two accents exist to
mean something in a given figure (e.g. two series, Re/Im), never to
decorate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


# Discrete stroke-weight levels for Curve.width_scale. A free scalar
# drifts to 1.0 across a corpus and flattens the ink hierarchy; three
# named levels keep the background-family / content / foreground-actor
# contrast the grammar asks for (actor:bg > 3:1). Off-level values stay
# legal — these are the defaults the eye is calibrated to, not a gate.
WEIGHT_BG = 0.55       # background family: the population behind the claim
WEIGHT_CONTENT = 1.0   # ordinary content ink
WEIGHT_ACTOR = 1.8     # THE object: one distinguished member, not many


class Role(Enum):
    CONTENT = auto()
    CONSTRUCTION = auto()
    ANNOTATION = auto()
    FRAME = auto()
    ACCENT1 = auto()
    ACCENT2 = auto()
    MUTED = auto()


@dataclass(frozen=True)
class Ink:
    color: str
    width: float          # px at reference canvas width
    dash: str | None = None  # SVG dasharray or None for solid


@dataclass(frozen=True)
class Style:
    inks: dict[Role, Ink]
    label_size_pt: float = 11.0
    point_radius: float = 3.0
    arrowhead_len: float = 10.0
    arrowhead_halfwidth: float = 3.6
    background: str = "white"
    # The theme's paper stock, its grain, and label casings ride the render by
    # DEFAULT: the figure is a printed page, and the contrast gate only earns
    # its keep on a ground that is actually there. Opt out with
    # `transparent_variant(theme)` or `figcheck --transparent` for ink on alpha
    # — no paper rect and no casing, since a casing painted in the background
    # colour is an opaque blob once the ground is gone. Grain is ink, not
    # paper: it renders either way.
    transparent: bool = False
    # casing thickness in px added on EACH side of a haloed label's glyphs
    # or a cased curve's stroke
    halo_width: float = 2.4
    # named dash levels: dash as identity, orthogonal to role
    dash_patterns: dict[str, str] = field(default_factory=lambda: {
        "solid": "", "dashed": "6 4.5", "dotted": "0.1 4",
        "fine-dashed": "4 3.5"})
    # cumulative scaled() factor; explicit per-item size_pt (authored in
    # reading-size points) multiplies through label_pt()
    type_scale: float = 1.0

    def ink(self, role: Role) -> Ink:
        return self.inks[role]

    def label_pt(self, explicit: float | None) -> float:
        """Resolve a label's size: explicit sizes are reading-size pt and
        scale with the format's ink scale, like the default."""
        return explicit * self.type_scale if explicit else self.label_size_pt

    def dash(self, spec: str) -> str | None:
        """Resolve a semantic dash name, or scale a raw dasharray by
        type_scale — named patterns scale through scaled(); raw specs must
        not silently escape the same ink scaling."""
        if spec in self.dash_patterns:
            return self.dash_patterns[spec] or None
        if self.type_scale != 1.0:
            return " ".join(f"{float(v) * self.type_scale:g}"
                            for v in spec.split()) or None
        return spec

    def hidden_variant(self, role: Role) -> tuple[str, float, float]:
        """(dash, opacity multiplier, width_scale multiplier) for a hidden
        run of `role` ink — the single source for hidden-line styling. Dash
        is the primary hiddenness signal; the multipliers step the ink
        toward MUTED but must keep it above the stroke contrast floor
        (asserted per theme in tests)."""
        return ("dashed", 0.85, 0.9)

    def scaled(self, k: float) -> "Style":
        """Every absolute quantity — type, stroke, heads, dots, dash
        periods — multiplied by k. Formats read below their declared
        width use this to keep ink at reading-size (format.ink_scale);
        proportions within the figure are untouched."""
        if k == 1.0:
            return self
        from dataclasses import replace

        def _dash(d: str) -> str:
            return " ".join(f"{float(v) * k:g}" for v in d.split())

        return replace(
            self,
            inks={r: Ink(i.color, i.width * k, _dash(i.dash) if i.dash else None)
                  for r, i in self.inks.items()},
            label_size_pt=self.label_size_pt * k,
            point_radius=self.point_radius * k,
            arrowhead_len=self.arrowhead_len * k,
            arrowhead_halfwidth=self.arrowhead_halfwidth * k,
            halo_width=self.halo_width * k,
            dash_patterns={name: _dash(d) if d else d
                           for name, d in self.dash_patterns.items()},
            type_scale=self.type_scale * k,
        )


DEFAULT_STYLE = Style(
    inks={
        Role.CONTENT: Ink("#1a1a1a", 1.7),
        Role.CONSTRUCTION: Ink("#8a8a8a", 1.1, dash="5 4"),
        Role.ANNOTATION: Ink("#1a1a1a", 1.1),
        Role.FRAME: Ink("#b8b8b8", 0.9, dash="1.5 3.5"),
        Role.ACCENT1: Ink("#3d6fb4", 1.9),
        Role.ACCENT2: Ink("#d1495b", 1.9),
        Role.MUTED: Ink("#b0b0b0", 1.3),
    }
)
