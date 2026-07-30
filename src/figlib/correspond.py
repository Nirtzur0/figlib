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
from .style import Role, Style

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
# Split is checked within each panel AND across panel pairs. It runs with no
# declaration, like every other gate — an undeclared composite is exactly the
# case where the reader has no legend to fall back on. The one thing it steps
# around is a panel pair a declared `Correspondence` already covers: there the
# residual above reports the same drift (`_facet` puts color in the identity
# fingerprint) with the fix the author actually needs — repaint, or add the key
# to `changes=`. Reporting it twice would make that fix ambiguous. A key
# declared in `changes=` is covered for the same reason: the correspondence has
# taken responsibility for it.
#
# Collision is pooled across the whole figure, because correspondence is
# exactly the thing that has to survive the panel boundary.
#
# Participation is by CLAIM, not by type. An item participates if it carries an
# explicit `.color` — which is a deliberate hue choice wherever it appears,
# including on a label, and a colored label is precisely where hue-as-noun gets
# read — or if it wears one of the object-bearing roles below. Scaffolding ink
# (CONSTRUCTION / ANNOTATION / FRAME) is not a hue claim: it is the ink that
# recedes, shared by everything wearing the role, and a keyed group routinely
# spans an object plus its annotation (demo_panels_zsquared keys a Curve, an
# AngleMark and an ANNOTATION label together).
_HUE_ROLES = frozenset({Role.CONTENT, Role.ACCENT1, Role.ACCENT2, Role.MUTED})


def _hue_marks(scene: Scene, style: Style):
    """(key, resolved color) for each keyed hue claim, render's precedence."""
    for item in scene.items:
        key = getattr(item, "key", None)
        if not key:
            continue
        color = getattr(item, "color", None)
        if color is None:
            role = getattr(item, "role", None)
            if role not in _HUE_ROLES:
                continue
            color = style.ink(role).color
        if isinstance(color, str) and color:
            yield key, color


def _panel_inks(scene: Scene, style: Style) -> dict[str, set[str]]:
    inks: dict[str, set[str]] = {}
    for key, color in _hue_marks(scene, style):
        inks.setdefault(key, set()).add(str(color).lower())
    return inks


def _covered_pairs(corrs: Iterable[Correspondence]) -> set[tuple[int, int]]:
    """Panel pairs the residual already walks, for every key they share."""
    pairs: set[tuple[int, int]] = set()
    for corr in corrs:
        parts = sorted(corr.parts)
        for a in range(len(parts)):
            for b in range(a + 1, len(parts)):
                pairs.add((parts[a], parts[b]))
    return pairs


def hue_binding_violations(scenes: Scene | Sequence[Scene], style: Style,
                           corrs: Iterable[Correspondence] = (),
                           ) -> list[Diagnostic]:
    """One hue, one named object — the reader's index into a composite.

    `scenes` is one Scene or a figure's panels in order. `corrs` are the
    declared correspondences, if any: the panel pairs they cover belong to
    the residual, and only those are stepped around.
    """
    from .theme import CORRESPONDENCE

    parts = [scenes] if isinstance(scenes, Scene) else list(scenes)
    per_panel = [_panel_inks(sc, style) for sc in parts]
    covered = _covered_pairs(corrs)
    multi = len(parts) > 1
    diags: list[Diagnostic] = []
    split_within: set[str] = set()

    def split(where: str, key: str, drawn: str, n: int) -> Diagnostic:
        return Diagnostic("hue-split", (
            f"{where}key {key!r} is one object drawn in {n} inks — {drawn}. "
            f"A hue names an object; two hues on one name read as two "
            f"objects. Give every mark of {key!r} the same color, or split "
            f"the key."))

    for i, inks in enumerate(per_panel):
        where = f"panel[{i}] " if multi else ""
        for key, seen in sorted(inks.items()):
            if len(seen) > 1:
                split_within.add(key)
                diags.append(split(where, key, ", ".join(sorted(seen)), len(seen)))

    # Across panels: the same key wearing different ink in two panels no
    # declared correspondence binds.
    for key in sorted({k for inks in per_panel for k in inks} - split_within):
        drifts = [(i, j) for i in range(len(parts)) for j in range(i + 1, len(parts))
                  if (i, j) not in covered
                  and key in per_panel[i] and key in per_panel[j]
                  and per_panel[i][key] != per_panel[j][key]]
        if not drifts:
            continue
        shown = sorted({i for pair in drifts for i in pair})
        drawn = ", ".join(f"panel[{i}] {'/'.join(sorted(per_panel[i][key]))}"
                          for i in shown)
        diags.append(Diagnostic("hue-split", (
            f"key {key!r} changes ink across panels — {drawn}. The hue is how "
            f"the reader carries the object over the panel boundary; recolour "
            f"so {key!r} is one hue everywhere, or declare the pair in a "
            f"Correspondence and let the residual name what varies.")))

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
