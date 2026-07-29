"""Composite figures: the declared binding, and the residual it leaks.

A composite figure's claim is never a picture. It is a predicate over a
*correspondence* — the reader is shown the same object twice and asked to
read what survived, or what changed. Three slots decide whether that
clicks, and only the third is mechanical:

1. **The binding** — which marks in one part ARE the same object as marks
   in another. Declared by naming them: items that share a `key` are the
   same object. Not a side table of panel indices; a name written where
   the geometry is built, so the two halves cannot drift apart in the
   source the way duplicated literals do.
2. **The declared variation** — `varies`, one sentence, plus `changes`,
   the keys it is allowed to move. Prose is unchecked here on purpose;
   the cold reader checks it (readback), because no gate can.
3. **The residual** — every OTHER difference across the binding. This is
   the whole reason composites fail: two things changed, so the diff is
   unattributable, and the reader cannot tell which one carries the
   claim. It is computable, so it is computed.

Deliberately NOT a taxonomy. Before/after, proof-by-transport,
instance-in-family, probe-object, analogy-to-a-familiar-picture are all
settings of (binding, variation) and none of them is a type here — naming
the patterns would fix the vocabulary at today's corpus. Name the axes;
let the patterns be what clusters in the declarations.

What the fingerprint deliberately excludes: **position**. A composite
whose parts agreed on position would have nothing to say — moving is
usually the claim. And **opacity**, which is a legibility channel (the
ghosted pre-image) rather than an identity one; folding it in would push
authors to stuff `changes` with things the claim does not turn on.

Scope: multi-panel `Figure`s. Single-scene composites — a ghost
pre-image, a word-scale inset — bind within one Scene, and identifying
their parts needs a selector this module does not have yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .figure import Figure, layout_figure
from .gates import Diagnostic
from .scene import Scene
from .style import Style

# Page scales this far apart are the same scale (rounding in slot widths).
SCALE_TOL = 0.005


@dataclass(frozen=True)
class Correspondence:
    """A declared relation between parts of a composite figure.

    `parts` are panel indices. `varies` is the one thing the binding does
    not preserve, in prose. `changes` are the keys it is allowed to move —
    every other keyed object must be identical across the parts, and a key
    listed here that does NOT move is a defect: the figure claims a
    difference it never draws.

    `frame` is "shared" (the default — the parts are read against each
    other, so their page scales must match) or the reason for a rescale,
    which exempts it and puts the admission in the source.
    """

    parts: tuple[int, ...]
    varies: str
    changes: tuple[str, ...] = ()
    frame: str = "shared"

    def __post_init__(self) -> None:
        if len(self.parts) < 2:
            raise ValueError("a correspondence needs at least two parts")
        if len(set(self.parts)) != len(self.parts):
            raise ValueError(f"repeated part in {self.parts}")
        if not self.varies.strip():
            raise ValueError("varies= must say what the binding does not preserve")


# One item's appearance, position excluded. A key's fingerprint in a part
# is the SET of these over its items, so "one curve" and "the same curve
# drawn in two arcs" are the same object.
Facet = str


def _facet(item: object) -> Facet:
    color = getattr(item, "color", None)
    dash = getattr(item, "dash", None)
    role = getattr(item, "role", None)
    text = getattr(item, "latex", None) or getattr(item, "label", None)
    parts = [f"type={type(item).__name__}",
             f"role={role.name if role is not None else '-'}",
             f"color={color or '-'}",
             f"dash={dash or '-'}"]
    # filled/hollow is a convention that carries meaning (stable vs unstable
    # fixed point, image vs pre-image vector), so it is identity, not style.
    if hasattr(item, "filled"):
        parts.append(f"filled={item.filled}")
    if text:
        parts.append(f"text={text}")
    return " ".join(parts)


def keyed(key: str, *items):
    """Tag every item with `key` and hand them back, for the producers that
    emit a group at once (`plots.series`, `plots.axis`, `surface3d.compose`).

    Threading `key=` through every emitter signature would be the same
    parameter written forty times; naming the group at the call site says
    the same thing where the meaning is:

        s.add(*keyed("branch", *plots.series(r, x_stable)))
    """
    for it in items:
        it.key = key
    return items


def _fingerprints(scene: Scene) -> dict[str, frozenset[Facet]]:
    out: dict[str, set[Facet]] = {}
    for item in scene.items:
        key = getattr(item, "key", None)
        if key:
            out.setdefault(key, set()).add(_facet(item))
    return {k: frozenset(v) for k, v in out.items()}


def _render(fp: frozenset[Facet] | None) -> str:
    if fp is None:
        return "absent"
    return " | ".join(sorted(fp))


def _key_diagnostics(corr: Correspondence,
                     prints: Sequence[dict[str, frozenset[Facet]]],
                     ) -> list[Diagnostic]:
    keys = sorted({k for p in prints for k in p})
    if not keys:
        return [Diagnostic("undeclared-binding", (
            f"parts {list(corr.parts)} declare a correspondence "
            f"(varies: {corr.varies}) but no item carries a key — the binding "
            f"exists only in prose. Give the marks that are the same object "
            f"the same key=."))]

    diags: list[Diagnostic] = []
    for key in keys:
        seen = [p.get(key) for p in prints]
        differs = len(set(seen)) > 1
        declared = key in corr.changes
        if differs and not declared:
            where = "  ".join(f"part[{i}] {_render(fp)}"
                              for i, fp in zip(corr.parts, seen))
            diags.append(Diagnostic("residual", (
                f"key {key!r} is not the same across the binding — {where}. "
                f"The declared variation is {corr.varies!r}; this is a second "
                f"difference, so the reader cannot attribute either. Make the "
                f"parts agree, or add {key!r} to changes= if it IS the claim.")))
        elif declared and not differs:
            diags.append(Diagnostic("stale-change", (
                f"key {key!r} is declared in changes= but is identical in every "
                f"part ({_render(seen[0])}) — the figure states a difference it "
                f"does not draw.")))
    for key in corr.changes:
        if key not in keys:
            diags.append(Diagnostic("stale-change", (
                f"changes= names key {key!r}, which no item in parts "
                f"{list(corr.parts)} carries.")))
    return diags


def _frame_diagnostics(corr: Correspondence, scales: Sequence[tuple[float, float]],
                       bound: Sequence[str]) -> list[Diagnostic]:
    """Page-scale parity across the binding.

    A declared rescale (`frame=` set to a reason) exempts the general
    comparison of lengths — but never the objects the figure declared FIXED.
    An invariant drawn at two page sizes is the invariance claim contradicted
    by the drawing, and it is the one thing `frame=` must not buy silence on.
    """
    (sx0, sy0), ref = scales[0], corr.parts[0]
    diags = []
    for part, (sx, sy) in zip(corr.parts[1:], scales[1:]):
        rx, ry = sx / sx0, sy / sy0
        if abs(rx - 1.0) <= SCALE_TOL and abs(ry - 1.0) <= SCALE_TOL:
            continue
        if corr.frame == "shared":
            diags.append(Diagnostic("frame-drift", (
                f"parts [{ref}] and [{part}] share a frame but not a page "
                f"scale: x x{rx:.4g}, y x{ry:.4g} "
                f"({sx0:.4g},{sy0:.4g} vs {sx:.4g},{sy:.4g} px per math unit). "
                f"Every length the reader compares across them is off by that "
                f"factor. Match the panels' lims and slot widths, or state the "
                f"rescale in frame=.")))
        elif bound:
            diags.append(Diagnostic("fixed-set-rescaled", (
                f"parts [{ref}] and [{part}] are at page scales x{rx:.4g}/"
                f"x{ry:.4g}, and {', '.join(repr(k) for k in bound)} "
                f"{'is' if len(bound) == 1 else 'are'} bound — declared "
                f"unchanged by {corr.varies!r} — so the figure draws its own "
                f"invariants at different sizes and the reader sees them "
                f"change. frame={corr.frame!r} exempts length comparison in "
                f"general; it cannot exempt the fixed set. Either match the "
                f"scales, or mark the rescale ON the page (a shared-size "
                f"reference or a stated magnification) so the size change "
                f"reads as the frame's, not the object's.")))
    return diags


# --- hue as a referential noun ----------------------------------------------
#
# corpus-study.md §6: a hue is a NOUN. It names one object, is declared once,
# and means that object everywhere it appears — which is what lets a reader
# cross a panel boundary without re-reading the legend. Two ways to break it,
# both mechanical:
#
#   hue-split      one key, two inks — the object appears twice as two objects
#   hue-collision  two keys, one correspondence hue — one noun, two referents
#
# Split is scoped WITHIN a Scene on purpose. Across a Figure's panels the same
# defect is already the residual above (`_facet` puts color in the identity
# fingerprint), and reporting it twice would make the fix ambiguous.
# Collision is pooled across the whole figure, because correspondence is
# exactly the thing that has to survive the panel boundary.
#
# Only marks participate. A keyed group routinely spans the object AND its
# annotation — demo_panels_zsquared keys a Curve, an AngleMark and a label
# together — and annotation ink is text ink, not a claim about the object's
# hue. Including it would report every properly-labelled object as split.
_HUE_BEARING = ("Curve", "FilledCurve", "Vector", "Point")


def _hue_marks(scene: Scene, style: Style):
    """(key, resolved color) for each keyed mark, render's own precedence."""
    for item in scene.items:
        key = getattr(item, "key", None)
        if not key or type(item).__name__ not in _HUE_BEARING:
            continue
        role = getattr(item, "role", None)
        color = getattr(item, "color", None)
        if color is None and role is not None:
            color = style.ink(role).color
        if color:
            yield key, color


def hue_binding_violations(scenes: Scene | Sequence[Scene],
                           style: Style) -> list[Diagnostic]:
    """One hue, one named object — the reader's index into a composite.

    `scenes` is one Scene or a figure's panels in order.
    """
    from .theme import CORRESPONDENCE

    parts = [scenes] if isinstance(scenes, Scene) else list(scenes)
    diags: list[Diagnostic] = []

    for i, scene in enumerate(parts):
        inks: dict[str, set[str]] = {}          # key -> lowered colors
        for key, color in _hue_marks(scene, style):
            inks.setdefault(key, set()).add(str(color).lower())
        where = f"panel[{i}] " if len(parts) > 1 else ""
        for key, seen in sorted(inks.items()):
            if len(seen) > 1:
                drawn = ", ".join(sorted(seen))
                diags.append(Diagnostic("hue-split", (
                    f"{where}key {key!r} is one object drawn in "
                    f"{len(seen)} inks — {drawn}. A hue names an object; two "
                    f"hues on one name read as two objects. Give every mark "
                    f"of {key!r} the same color, or split the key.")))

    # Collision: pooled, and only the categorical channel. A bare '#rrggbb'
    # is not claiming identity, and ramp/shade hues are ordered quantity.
    owners: dict[str, set[str]] = {}
    for scene in parts:
        for key, color in _hue_marks(scene, style):
            if getattr(color, "channel", None) == CORRESPONDENCE:
                owners.setdefault(str(color).lower(), set()).add(key)
    for color, keys in sorted(owners.items()):
        if len(keys) > 1:
            named = ", ".join(repr(k) for k in sorted(keys))
            diags.append(Diagnostic("hue-collision", (
                f"correspondence hue {color} names {len(keys)} objects — "
                f"{named}. The hue is the reader's index into the figure, so "
                f"one hue must mean one thing everywhere. Give each object "
                f"its own categorical slot, or key them as the same object.")))
    return diags


def residual(fig: Figure, corrs: Iterable[Correspondence], style: Style,
             width_px: float) -> list[Diagnostic]:
    """Everything that differs across each declared binding and was not
    declared. Empty means the composite holds exactly one variable."""
    lay = layout_figure(fig, width_px)
    diags: list[Diagnostic] = []
    for corr in corrs:
        for i in corr.parts:
            if not 0 <= i < len(fig.panels):
                raise IndexError(
                    f"correspondence part {i} but the figure has "
                    f"{len(fig.panels)} panels")
        prints = [_fingerprints(fig.panels[i].scene) for i in corr.parts]
        scales = [(lay.slots[i].transform.scale_x, lay.slots[i].transform.scale_y)
                  for i in corr.parts]
        bound = sorted({k for p in prints for k in p} - set(corr.changes))
        diags += _key_diagnostics(corr, prints)
        diags += _frame_diagnostics(corr, scales, bound)
    return diags
