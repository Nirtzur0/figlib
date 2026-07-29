"""The ink-free search must stay inside what the reader can actually see.

A label moved to a suggestion outside a clip="frame" rect PASSES every gate
and renders invisible: the clip removes it, and no gate measures drawn ink.
That silent-failure mode is what these tests pin.
"""

from __future__ import annotations

import numpy as np

from figlib.gates import _nearest_free_center, mechanical
from figlib.layout import Transform
from figlib.scene import Curve, MathLabel, Scene
from figlib.style import DEFAULT_STYLE, Role


def test_nearest_free_center_respects_bounds():
    box = (10.0, 10.0, 40.0, 24.0)
    bounds = (100.0, 100.0, 300.0, 260.0)
    c = _nearest_free_center(box, [], [], 600.0, 400.0, bounds=bounds)
    assert c is not None
    w, h = box[2] - box[0], box[3] - box[1]
    x0, y0, x1, y1 = c[0] - w / 2, c[1] - h / 2, c[0] + w / 2, c[1] + h / 2
    assert bounds[0] <= x0 and bounds[1] <= y0
    assert x1 <= bounds[2] and y1 <= bounds[3]


def test_without_bounds_the_whole_canvas_is_fair_game():
    """Unclipped scenes keep the old behaviour — the margin band is legitimate
    real estate when nothing is clipping it away. The same box that the bounded
    case pushes out to (>=100, >=100) stays near its own corner here."""
    box = (10.0, 10.0, 40.0, 24.0)
    c = _nearest_free_center(box, [], [], 600.0, 400.0)
    assert c is not None
    assert c[0] < 100.0 and c[1] < 100.0


def _clipped_scene() -> Scene:
    th = np.linspace(0, 2 * np.pi, 400)
    s = Scene(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), clip="frame")
    for r in np.linspace(0.1, 1.15, 22):
        s.add(Curve(np.column_stack([r * np.cos(th), r * np.sin(th)]),
                    role=Role.CONTENT))
    s.add(MathLabel(r"\theta", anchor=(0.0, 0.0), role=Role.ANNOTATION))
    return s


def test_a_clipped_scene_never_suggests_a_point_it_would_clip_away():
    scene = _clipped_scene()
    t = Transform(scene, width_px=680)
    fx0, fy0 = t.to_canvas((scene.xlim[0], scene.ylim[1]))
    fx1, fy1 = t.to_canvas((scene.xlim[1], scene.ylim[0]))
    diags = [d for d in mechanical(scene, DEFAULT_STYLE, width_px=680)
             if d.kind == "label-on-ink"]
    for d in diags:
        if "math coords" not in d.detail:
            continue
        mx, my = (float(v) for v in
                  d.detail.split("center (")[1].split(")")[0].split(", "))
        cx, cy = t.to_canvas((mx, my))
        assert fx0 - 1 <= cx <= fx1 + 1, f"suggested x outside frame: {d.detail}"
        assert fy0 - 1 <= cy <= fy1 + 1, f"suggested y outside frame: {d.detail}"
