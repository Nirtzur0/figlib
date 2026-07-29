"""Formats: the figure's slot on the page.

The sizing invariant: canvas units ARE display CSS pixels. A figure
declares the width of the slot it will occupy (module-level FORMAT), and
every absolute quantity — label pt, stroke px, arrowhead px, dot radius —
is thereby at its final rendered size. Type is never scaled to fit; the
figure is designed at the size it will be read.

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


# A wrapped/side figure, half a text column.
MARGIN = Format("margin", 340)
# The default: figure spans a prose column (Distill-style article text).
COLUMN = Format("column", 680)
# Breaks out of the column for dense or 3D content.
WIDE = Format("wide", 1000)
