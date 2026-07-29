"""House style: semantic roles resolved to ink.

The grammar (from Needham): solid = content, dashed = construction,
dotted = frame. Color is semantic or absent — the two accents exist to
mean something in a given figure (e.g. two series, Re/Im), never to
decorate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


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

    def ink(self, role: Role) -> Ink:
        return self.inks[role]


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
