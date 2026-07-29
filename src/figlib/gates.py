"""Deterministic gates: numerical assertions and mechanical render checks.

The readback gate (cold-reader test) lives in readback.py — it needs a
model; these two do not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .layout import Transform
from .scene import MathLabel, Scene
from .style import Style
from .typeset import render_math

# Fraction of a label's height assumed to hang below the baseline.
_DESCENT_FRAC = 0.2


@dataclass(frozen=True)
class Diagnostic:
    kind: str    # 'label-collision' | 'clipped' | 'numerical'
    detail: str


def _label_boxes(scene: Scene, style: Style, t: Transform) -> list[tuple[str, tuple[float, float, float, float]]]:
    """Canvas-space (x0, y0, x1, y1) per MathLabel."""
    boxes = []
    for it in scene.items:
        if not isinstance(it, MathLabel):
            continue
        m = render_math(it.latex, it.size_pt or style.label_size_pt)
        w, h = m.width_px, m.height_px
        x, y = t.to_canvas(it.anchor)
        x += it.offset_px[0]
        y += it.offset_px[1]
        if it.ha == "center":
            x -= w / 2
        elif it.ha == "right":
            x -= w
        if it.va == "base":
            y0 = y - h * (1 - _DESCENT_FRAC)
        elif it.va == "top":
            y0 = y
        elif it.va == "center":
            y0 = y - h / 2
        else:  # bottom
            y0 = y - h
        boxes.append((it.latex, (x, y0, x + w, y0 + h)))
    return boxes


def _overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def mechanical(scene: Scene, style: Style, width_px: float = 900) -> list[Diagnostic]:
    t = Transform(scene, width_px=width_px)
    boxes = _label_boxes(scene, style, t)
    diags: list[Diagnostic] = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if _overlap(boxes[i][1], boxes[j][1]):
                diags.append(Diagnostic(
                    "label-collision",
                    f"'{boxes[i][0]}' overlaps '{boxes[j][0]}'"))
    for latex, (x0, y0, x1, y1) in boxes:
        if x0 < 0 or y0 < 0 or x1 > t.canvas_w or y1 > t.canvas_h:
            diags.append(Diagnostic("clipped", f"'{latex}' extends outside the canvas"))
    return diags


def numerical(assert_fn: Callable[[], None]) -> list[Diagnostic]:
    try:
        assert_fn()
    except AssertionError as e:
        return [Diagnostic("numerical", str(e) or "assertion failed")]
    return []
