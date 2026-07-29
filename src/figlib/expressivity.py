"""Floor-side expressivity signals: making under-expression measurable.

Every gate in gates.py is a ceiling — too small, too crowded, too faint.
A one-sided oracle lets the iteration loop converge to the sparsest
figure that passes, which is exactly the over-simplicity failure mode.
These signals are the floor side: ink fraction, stroke-weight hierarchy,
where the ink mass sits, which channels went unused, how large the
largest curve family is.

They are ADVISORY — they ride Report.signals, never diagnostics, and can
never fail a figure. A margin sketch is allowed to be sparse; the loop
just has to see that it is, instead of reading silence as adequacy.

All areas are approximations (stroke length x width, shoelace fills,
glyph-coverage labels). The numbers only need to be monotone in ink to
steer iteration; they are not measurements of the rendered raster.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

from .layout import Transform
from .scene import (Brace, Callout, Curve, FilledCurve, MathLabel, Point,
                    RasterField, Scene, Vector)
from .style import Role, Style

# Below this fraction of canvas the figure reads as an outline of itself.
SPARSE_INK_FRAC = 0.04
# Content strokes at fewer distinct weight levels than this, or spanning
# less than this ratio, carry no hierarchy (grammar: bg/content/actor).
MIN_WEIGHT_LEVELS = 2
MIN_WEIGHT_RATIO = 1.5
# Flat-hierarchy advice only fires once there are enough strokes to rank.
MIN_STROKES_FOR_HIERARCHY = 4

# Fraction of a label bbox actually covered by glyph ink.
_GLYPH_COVERAGE = 0.30
# Mean coverage assumed for a raster field (unknown without rendering).
_RASTER_COVERAGE = 0.6
# Ink-equivalent opacity of a pattern fill (texture, not a full wash).
_PATTERN_COVERAGE = 0.15

_SCAFFOLD = frozenset({Role.CONSTRUCTION, Role.FRAME, Role.MUTED})
_CONTENT_ROLES = frozenset({Role.CONTENT, Role.ACCENT1, Role.ACCENT2})


def _polyline_len(pts: np.ndarray) -> float:
    d = np.diff(np.asarray(pts, dtype=float), axis=0)
    return float(np.hypot(d[:, 0], d[:, 1]).sum())


def _shoelace(pts: np.ndarray) -> float:
    p = np.asarray(pts, dtype=float)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)))


def _dash_duty(dash: str | None, style: Style) -> float:
    """Fraction of a dashed stroke that is ink (SVG dasharray: on off ...)."""
    if not dash:
        return 1.0
    resolved = style.dash(dash)
    if not resolved:
        return 1.0
    vals = [float(v) for v in resolved.split()]
    if not vals or sum(vals) <= 0:
        return 1.0
    on = sum(vals[0::2])
    return max(min(on / sum(vals), 1.0), 0.05)


def _curve_width_px(it: Curve, style: Style) -> float:
    w = style.ink(it.role).width * it.width_scale
    if it.width_profile:
        w *= sum(it.width_profile) / len(it.width_profile)
    return w


class _Tally:
    """Ink areas by (type, role), channel usage, weights, families."""

    def __init__(self) -> None:
        self.ink: dict[tuple[str, str], float] = defaultdict(float)
        self.channels: set[str] = set()
        self.weights: list[float] = []          # content-role stroke widths, px
        self.families: Counter = Counter()      # same-styled curve groups
        self.canvas_area = 0.0

    def add_scene(self, scene: Scene, style: Style, t: Transform) -> None:
        for it in scene.items:
            self._add_item(it, style, t)

    def _note_hue(self, color) -> None:
        ch = getattr(color, "channel", None)
        if ch:
            self.channels.add(ch)

    def _add_item(self, it, style: Style, t: Transform) -> None:
        if isinstance(it, Curve):
            length = _polyline_len(t.to_canvas_arr(it.pts))
            w = _curve_width_px(it, style)
            duty = 1.0 if it.width_profile else _dash_duty(it.dash, style)
            self.ink[("Curve", it.role.name)] += length * w * duty
            if it.role in _CONTENT_ROLES:
                self.weights.append(round(w, 1))
            self.families[(it.role.name, str(it.color), it.dash,
                           round(it.width_scale, 2))] += 1
            if it.dash or style.ink(it.role).dash:
                self.channels.add("dash")
            if it.arrows:
                self.channels.add("arrows")
            if it.width_profile:
                self.channels.add("width_profile")
            if it.casing:
                self.channels.add("casing")
            if it.opacity < 1.0:
                self.channels.add("opacity")
            self._note_hue(it.color)
        elif isinstance(it, FilledCurve):
            area = _shoelace(t.to_canvas_arr(it.pts))
            for hole in it.holes:
                area -= _shoelace(t.to_canvas_arr(hole))
            cover = _PATTERN_COVERAGE if it.pattern else it.opacity
            self.ink[("FilledCurve", it.role.name)] += max(area, 0.0) * cover
            self.channels.add("fill")
            if it.pattern:
                self.channels.add("pattern")
            if it.holes:
                self.channels.add("holes")
            self._note_hue(it.color)
        elif isinstance(it, Vector):
            (x0, y0), (x1, y1) = t.to_canvas(it.tail), t.to_canvas(it.tip)
            w = style.ink(it.role).width * it.width_scale
            length = math.hypot(x1 - x0, y1 - y0)
            head = style.arrowhead_len * style.arrowhead_halfwidth
            self.ink[("Vector", it.role.name)] += length * w + head
            if it.role in _CONTENT_ROLES:
                self.weights.append(round(w, 1))
        elif isinstance(it, Point):
            r = style.point_radius * it.radius_scale
            self.ink[("Point", it.role.name)] += math.pi * r * r
        elif isinstance(it, RasterField):
            x0, y0 = t.to_canvas((it.extent[0], it.extent[2]))
            x1, y1 = t.to_canvas((it.extent[1], it.extent[3]))
            area = abs(x1 - x0) * abs(y1 - y0)
            self.ink[("RasterField", "-")] += area * it.opacity * _RASTER_COVERAGE
            self.channels.add("raster")
            if it.ramp is not None:
                self._note_hue(it.ramp(0.5))
        elif isinstance(it, (Brace, Callout)):
            # small annotation glyphs; their MathLabel area is counted by
            # the label pass in signals()
            self.channels.add("glyphs")

    def add_label_area(self, area: float) -> None:
        self.ink[("MathLabel", "-")] += area * _GLYPH_COVERAGE


# Channels worth calling out when absent — each has sat unused corpus-wide
# while being the honest encoding for a class of claims.
_NOTABLE = ("raster", "fill", "wash", "width_profile", "pattern", "casing",
            "dash", "arrows")


def _lines(tly: _Tally) -> list[str]:
    out: list[str] = []
    total = sum(tly.ink.values())
    frac = min(total / tly.canvas_area, 1.0) if tly.canvas_area else 0.0

    heaviest = ""
    scaffold_flag = ""
    if tly.ink:
        (typ, role), amt = max(tly.ink.items(), key=lambda kv: kv[1])
        share = amt / total if total else 0.0
        heaviest = f" | heaviest: {typ} {role} {share:.0%}"
        if role in {r.name for r in _SCAFFOLD}:
            scaffold_flag = (
                f"heaviest ink is scaffolding ({typ} {role}) — the claim, "
                f"not the furniture, should carry the largest ink mass")

    levels = sorted(set(tly.weights))
    ratio = (levels[-1] / levels[0]) if len(levels) > 1 and levels[0] > 0 else 1.0
    weight_str = (f" | weights: {len(levels)} levels "
                  f"[{', '.join(f'{w:g}' for w in levels)}]px"
                  + (f" span {ratio:.1f}:1" if len(levels) > 1 else "")
                  if levels else "")

    fam = tly.families.most_common(1)
    fam_str = f" | family max n={fam[0][1]}" if fam else ""

    used = sorted(tly.channels)
    unused = [c for c in _NOTABLE if c not in tly.channels]
    chan_str = (f"channels: {','.join(used) if used else '-'}"
                + (f"; unused: {','.join(unused)}" if unused else ""))

    out.append(f"ink {frac:.1%} of canvas{heaviest}{weight_str}{fam_str}")
    out.append(chan_str)

    if frac < SPARSE_INK_FRAC:
        out.append(
            f"sparse: ink covers {frac:.1%} (< {SPARSE_INK_FRAC:.0%}) — "
            f"consider a denser family, a field layer (RasterField / wash), "
            f"or a smaller Format; emptiness passes every gate but argues "
            f"nothing")
    n_strokes = len(tly.weights)
    if (n_strokes >= MIN_STROKES_FOR_HIERARCHY
            and (len(levels) < MIN_WEIGHT_LEVELS or ratio < MIN_WEIGHT_RATIO)):
        out.append(
            f"flat weight hierarchy: {n_strokes} content strokes at "
            f"{len(levels)} level(s), span {ratio:.1f}:1 — use "
            f"WEIGHT_BG/CONTENT/ACTOR (0.55/1.0/1.8) so the claim's object "
            f"dominates the family behind it")
    if scaffold_flag:
        out.append(scaffold_flag)
    return out


def signals(built, style: Style, width_px: float) -> list[str]:
    """Expressivity report for a Scene or a multi-panel Figure."""
    from .gates import _figure_label_entries, _label_boxes
    from .figure import Figure, layout_figure

    tly = _Tally()
    if isinstance(built, Figure):
        lay = layout_figure(built, width_px)
        tly.canvas_area = lay.canvas_w * lay.canvas_h
        for panel, slot in zip(built.panels, lay.slots):
            tly.add_scene(panel.scene, style, slot.transform)
        entries, _, _ = _figure_label_entries(built, style, width_px)
        for (_, _, (x0, y0, x1, y1)), _owner in entries:
            tly.add_label_area((x1 - x0) * (y1 - y0))
    else:
        t = Transform(built, width_px=width_px)
        tly.canvas_area = t.canvas_w * t.canvas_h
        tly.add_scene(built, style, t)
        for _, _, (x0, y0, x1, y1) in _label_boxes(built, style, t):
            tly.add_label_area((x1 - x0) * (y1 - y0))
    return _lines(tly)
