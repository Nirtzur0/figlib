"""Formats: the figure's slot on the page.

The sizing invariant: canvas units ARE display CSS pixels. A figure
declares the width of the slot it will occupy (module-level FORMAT) —
or declares none, and the smallest format that carries its annotation
census is derived (derive_format). Every absolute quantity — label pt,
stroke px, arrowhead px, dot radius — is thereby at its final rendered
size. Type is never scaled to fit; the figure is designed at the size
it will be read.

Consequences:
  * 11 pt labels are the same physical size in every figure.
  * A figure that cannot fit its annotation at its declared format is a
    design failure (too much annotation or too small a slot), and the
    mechanical gate says so — the fix is a bigger format or less ink,
    never smaller type.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Format:
    name: str
    display_width_px: float   # CSS px of the page slot at 96 dpi
    # Formats wider than the prose column still get READ at roughly
    # column width (fit-to-page, standalone PNG viewers). ink_scale =
    # declared/reading width compensates: type, strokes, heads, dots all
    # scale so the figure is legible at its reading size, not its
    # declared size. 1.0 where declared == reading width.
    ink_scale: float = 1.0


# A wrapped/side figure, half a text column.
MARGIN = Format("margin", 340)
# The default: figure spans a prose column (Distill-style article text).
COLUMN = Format("column", 680)
# Breaks out of the column for dense or 3D content; read near column width.
WIDE = Format("wide", 1000, ink_scale=1.45)

# The ladder, smallest first.
FORMATS = (MARGIN, COLUMN, WIDE)


# --- deriving the format from the annotation census -------------------------
#
# The mechanism the ladder actually offers: MARGIN -> COLUMN quarters the
# load fraction (same ink, 4x the canvas area), but COLUMN -> WIDE is
# near-invariant BY DESIGN — WIDE's ink_scale (1.45) tracks its width
# ratio (1000/680), so canvas and ink grow together and annotation
# headroom barely changes. A larger format is therefore not an assumed
# fix; each candidate is *evaluated*. MARGIN is never derived — taking
# the margin slot is page-layout intent, not a sizing consequence.


def annotation_census(built, base_style, fmt: Format) -> tuple[float, float]:
    """(total label area, tallest label height) as fractions of the
    canvas this built Scene/Figure would occupy at fmt — the same boxes
    the mechanical gate measures, at fmt's ink scale."""
    from .figure import Figure
    from .gates import _figure_label_entries, _label_entries
    from .layout import Transform

    style = base_style.scaled(fmt.ink_scale)
    width = fmt.display_width_px
    if isinstance(built, Figure):
        entries, canvas_w, canvas_h = _figure_label_entries(built, style, width)
    else:
        t = Transform(built, width_px=width)
        entries = _label_entries(built, style, t)
        canvas_w, canvas_h = t.canvas_w, t.canvas_h
    area = 0.0
    tallest = 0.0
    for (_, _, (x0, y0, x1, y1)), _ in entries:
        area += (x1 - x0) * (y1 - y0)
        tallest = max(tallest, y1 - y0)
    return area / (canvas_w * canvas_h), tallest / canvas_h


def smallest_carrier(built, base_style,
                     ladder: tuple[Format, ...] = (COLUMN, WIDE)) -> Format | None:
    """The smallest format on the ladder whose canvas carries the
    annotation load, or None — meaning the load is a design defect at
    every size, and the fix is less annotation."""
    from .gates import MAX_ANNOTATION_AREA_FRAC, MAX_LABEL_HEIGHT_FRAC

    for fmt in ladder:
        area_frac, height_frac = annotation_census(built, base_style, fmt)
        if (area_frac <= MAX_ANNOTATION_AREA_FRAC
                and height_frac <= MAX_LABEL_HEIGHT_FRAC):
            return fmt
    return None


def derive_format(built, base_style) -> Format:
    """The format for a program that declares none: the smallest carrier
    at or above COLUMN, or COLUMN when nothing carries (render at the
    default and let the gate state the defect precisely)."""
    return smallest_carrier(built, base_style) or COLUMN
