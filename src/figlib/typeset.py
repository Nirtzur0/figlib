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


# fraction of the typeset height that sits below the baseline
DESCENT_FRAC = 0.2


# Typeface as a semantic channel: monospace = literal model input / data,
# sans = human interpretation, absent = mathematics. Not appearance — a
# theme never sees it, because swapping palettes must not change what a
# token strip CLAIMS to be.
_REGISTERS = {"mono": r"\mathtt", "sans": r"\mathsf"}

# Latex machinery: a character whose meaning is structural, so it has no
# per-character register. `\mathtt{^}` is not "a monospace caret", it is a
# syntax error waiting to render as garbage.
_MACHINERY = frozenset(r"\{}^_$&#%~")


def apply_register(latex: str, register: str | None) -> str:
    """Set `latex` in the register's typeface, ONE CHARACTER AT A TIME.

    Per atom, not per string, and that is not a style choice: ziamath's
    font commands reach exactly one atom, so `\\mathtt{tok}` silently
    typesets `tok` in serif italic — identical glyphs to no command at
    all. `\\mathtt{t}\\mathtt{o}\\mathtt{k}` is what actually switches the
    font. (`\\texttt` changes glyphs but does not monospace them, so it is
    not the fix either.) The residual inter-atom math spacing is the price.

    A register is therefore only defined on a LITERAL CHARACTER RUN — a
    token, a word, a filename. Anything containing latex machinery
    (backslash, braces, `^`, `_`, `$`, `&`, `#`, `%`, `~`) raises: a
    control sequence has no per-character reading, and mathematics is
    already the default register. Digits and punctuation pass through
    wrapped but unstyled, because STIX has no monospace digit reachable
    this way; letters carry the signal. A space becomes `\\ `.

    Apply this at EVERY point a label is measured or drawn — the
    mechanical gate's boxes are only on the ink if measurement and drawing
    typeset the same string, and a mono run is ~20% wider than the serif
    measurement of the same characters.
    """
    if register is None:
        return latex
    try:
        cmd = _REGISTERS[register]
    except KeyError:
        raise ValueError(f"unknown register {register!r}; have {sorted(_REGISTERS)}")
    bad = sorted(set(latex) & _MACHINERY)
    if bad:
        raise ValueError(
            f"cannot set {latex!r} in register {register!r}: it contains "
            f"{bad}, and this is a PER-CHARACTER register defined only on a "
            f"literal character run. Fix: drop the register — mathematics is "
            f"the default register — or split the label so the literal run "
            f"carries the register by itself.")
    return "".join("\\ " if c == " " else f"{cmd}{{{c}}}" for c in latex)


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


def label_box(latex: str, size_pt: float, x_px: float, y_px: float,
              halign: str = "left", valign: str = "base",
              ) -> tuple[float, float, float, float]:
    """Exact typeset bbox in canvas px for a label anchored at (x_px, y_px).

    One definition, used by both the casing that paints behind a label and
    the mechanical gate that checks what it collides with — they cannot
    disagree about where a label is.
    """
    m = render_math(latex, size_pt)
    w, h = m.width_px, m.height_px
    if halign == "center":
        x_px -= w / 2
    elif halign == "right":
        x_px -= w
    if valign == "base":
        y0 = y_px - h * (1 - DESCENT_FRAC)
    elif valign == "top":
        y0 = y_px
    elif valign == "center":
        y0 = y_px - h / 2
    else:  # bottom
        y0 = y_px - h
    return (x_px, y0, x_px + w, y0 + h)


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

    With halo_color set, ONE rect in that color is laid behind the whole
    label — its bbox grown by halo_width on each side. The casing used to be
    per-glyph (a stroked copy of the glyphs), which reads as a separate patch
    behind every letter and bites the ink underneath in chunks between them;
    the label is one object and its casing is one shape."""
    if halo_color is not None and halo_width > 0.0:
        x0, y0, x1, y1 = label_box(latex, size_pt, x_px, y_px, halign, valign)
        ET.SubElement(svg_root, "rect", {
            "class": "label-casing",
            "x": f"{x0 - halo_width:.3f}", "y": f"{y0 - halo_width:.3f}",
            "width": f"{x1 - x0 + 2 * halo_width:.3f}",
            "height": f"{y1 - y0 + 2 * halo_width:.3f}",
            "fill": halo_color})
    m = _latex(latex, size_pt, color)
    m.drawon(svg_root, x_px, y_px, halign=halign, valign=valign)
