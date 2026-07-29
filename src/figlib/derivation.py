"""Annotated derivation rows: the equation IS the figure.

The corpus device (docs/corpus-study.md, "annotated derivation"): each
term of an expansion carries a prose gloss anchored beneath it, and
typography does the layout. `derivation_row` places LaTeX terms
left-to-right at their REAL typeset widths (typeset.render_math — the same
metrics the mechanical gate boxes with), sets each operator midway in the
whitespace between its neighbors with `gap_pt` of clearance on both sides,
and hangs sans-register glosses beneath the glossed terms.

Layout happens in math units but is metric in px: the caller states
`px_per_unit`, canvas px per math unit — compute it from the figure's
FORMAT width and xlim via `schematic.px_per_unit`. (`plots.px_units`
serves the same bridge but returns the RECIPROCAL, math units per px;
invert it before passing it here.) Point sizes convert through `typeset.PX_PER_PT`
— `size_pt` is authored reading-size pt, the convention every MathLabel
already uses.

Every produced label is `pin=True`: position IS meaning here, and the
auto-place pass must not touch it. Colliding glosses are NOT solved
(doctrine: no constraint solvers) — the mechanical gate reports them and
the figure program shortens the prose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .scene import Curve, Item, MathLabel, XY
from .style import Role
from .typeset import PX_PER_PT, apply_register, render_math

_DEFAULT_SIZE_PT = 11.0     # style.Style.label_size_pt's default reading size
_GLOSS_SCALE = 0.8          # secondary voice: smaller than the mathematics
_TICK_FRAC = 0.6            # tick spans the middle 0.6 of the drop


@dataclass(frozen=True)
class Term:
    """One term of the expansion, with its optional prose gloss."""

    latex: str
    gloss: str | None = None
    role: Role = Role.CONTENT
    key: str | None = None


def derivation_row(
    terms: Sequence[Term],
    *,
    lhs: str | None = None,
    op: str = "+",
    y: float = 0.0,
    size_pt: float | None = None,
    gloss_size_pt: float | None = None,
    px_per_unit: float,
    gap_pt: float = 14.0,
    drop_factor: float = 1.8,
) -> tuple[list[Item], list[XY]]:
    """Lay out `lhs = term op term op ...` with glosses hung beneath.

    Returns `(items, anchors)`: the scene items, and the per-term center
    anchors (math coords) so the figure program can add more annotation —
    e.g. brace a group of terms with the existing `Brace`.

    `px_per_unit` is math units -> canvas px (see module docstring); all
    widths come from `typeset.render_math` at the resolved size, so terms
    can never overlap and the operators read as typeset mathematics, not
    boxes in a row. `gap_pt` is the clearance floor on EACH side of an
    operator; the operator sits midway in that whitespace. Glossed terms
    get a `Role.ANNOTATION`, `register="sans"` gloss centered `drop_factor
    x` the term's height below `y`, joined by a short vertical tick.
    """
    size = size_pt if size_pt is not None else _DEFAULT_SIZE_PT
    gsize = gloss_size_pt if gloss_size_pt is not None else _GLOSS_SCALE * size
    gap_u = gap_pt * PX_PER_PT / px_per_unit

    def measure(latex: str, pt: float, register: str | None = None):
        m = render_math(apply_register(latex, register), pt)
        return m.width_px / px_per_unit, m.height_px / px_per_unit

    def label(latex: str, x: float, **kw) -> MathLabel:
        return MathLabel(latex, (x, y), size_pt=size, ha="center",
                         va="center", pin=True, **kw)

    items: list[Item] = []
    anchors: list[XY] = []
    cursor = 0.0                       # left edge of the next element

    def emit_op(latex: str) -> None:
        nonlocal cursor
        w, _ = measure(latex, size)
        items.append(label(latex, cursor + gap_u + w / 2.0))
        cursor += 2.0 * gap_u + w

    if lhs is not None:
        w, _ = measure(lhs, size)
        items.append(label(lhs, cursor + w / 2.0))
        cursor += w
        emit_op("=")

    for i, t in enumerate(terms):
        if i > 0:
            emit_op(op)
        w, h = measure(t.latex, size)
        cx = cursor + w / 2.0
        items.append(label(t.latex, cx, role=t.role, key=t.key))
        anchors.append((cx, y))
        cursor += w

        if t.gloss is not None:
            drop = drop_factor * h
            items.append(MathLabel(
                t.gloss, (cx, y - drop), role=Role.ANNOTATION,
                size_pt=gsize, ha="center", va="center", pin=True,
                register="sans"))
            # short vertical tick centered in the drop span: from below the
            # term to above the gloss, never touching either
            pad = (1.0 - _TICK_FRAC) / 2.0 * drop
            items.append(Curve(
                np.array([[cx, y - pad], [cx, y - drop + pad]]),
                role=Role.ANNOTATION))

    return items, anchors
