"""expressivity.py: floor-side signals — measurement, never a gate failure.

The gate stack is one-sided (every check is a ceiling), so the iteration
loop converges to the sparsest figure that passes. These signals make
under-expression visible: ink fraction, weight hierarchy, heaviest ink
group, channel utilization, family size. They are advisory lines in the
report; a figure can never FAIL on them.
"""

import numpy as np

from figlib.expressivity import signals
from figlib.scene import Curve, FilledCurve, MathLabel, RasterField, Scene
from figlib.style import (DEFAULT_STYLE, WEIGHT_ACTOR, WEIGHT_BG,
                          WEIGHT_CONTENT, Role)


def _joined(scene):
    return "\n".join(signals(scene, DEFAULT_STYLE, width_px=680.0))


def _line(pts):
    return np.asarray(pts, dtype=float)


def test_sparse_scene_flagged():
    scene = Scene(xlim=(0, 10), ylim=(0, 10))
    scene.add(Curve(_line([(1, 1), (9, 9)])))
    out = _joined(scene)
    assert "sparse" in out
    assert "ink" in out


def test_dense_scene_not_flagged_sparse():
    scene = Scene(xlim=(0, 10), ylim=(0, 10))
    ys = np.linspace(0.5, 9.5, 40)
    for y in ys:
        scene.add(Curve(_line([(0.2, y), (9.8, y)]), width_scale=2.0))
    assert "sparse" not in _joined(scene)


def test_flat_weight_hierarchy_flagged():
    scene = Scene(xlim=(0, 10), ylim=(0, 10))
    for y in range(1, 7):
        scene.add(Curve(_line([(1, y), (9, y)])))   # all width_scale=1.0
    out = _joined(scene)
    assert "flat weight hierarchy" in out


def test_weight_levels_satisfy_hierarchy():
    scene = Scene(xlim=(0, 10), ylim=(0, 10))
    for y in range(1, 6):
        scene.add(Curve(_line([(1, y), (9, y)]), width_scale=WEIGHT_BG))
    scene.add(Curve(_line([(1, 7), (9, 7)]), width_scale=WEIGHT_CONTENT))
    scene.add(Curve(_line([(1, 8), (9, 8)]), width_scale=WEIGHT_ACTOR))
    out = _joined(scene)
    assert "flat weight hierarchy" not in out
    assert "weights: 3 levels" in out


def test_heaviest_scaffold_flagged():
    scene = Scene(xlim=(0, 10), ylim=(0, 10))
    box = _line([(0, 0), (10, 0), (10, 10), (0, 10)])
    scene.add(FilledCurve(box, role=Role.FRAME, opacity=1.0, outline=False))
    scene.add(Curve(_line([(1, 1), (2, 2)])))
    out = _joined(scene)
    assert "scaffolding" in out


def test_channel_utilization_reported():
    scene = Scene(xlim=(0, 1), ylim=(0, 1))
    scene.add(RasterField(np.ones((4, 4)), extent=(0, 1, 0, 1)))
    scene.add(Curve(_line([(0, 0), (1, 1)]), dash="dashed"))
    out = _joined(scene)
    assert "raster" in out
    assert "dash" in out
    # unused channels are named, so their absence is visible to the loop
    assert "unused" in out


def test_family_size_reported():
    scene = Scene(xlim=(0, 10), ylim=(0, 10))
    for y in np.linspace(1, 9, 10):
        scene.add(Curve(_line([(1, y), (9, y)]), width_scale=WEIGHT_BG))
    assert "family max n=10" in _joined(scene)


def test_labels_and_points_do_not_crash():
    scene = Scene(xlim=(0, 10), ylim=(0, 10))
    scene.add(MathLabel("x^2", (5, 5)))
    scene.add(Curve(_line([(1, 1), (9, 9)])))
    assert signals(scene, DEFAULT_STYLE, width_px=680.0)


def test_figure_variant_runs():
    from figlib.figure import Figure, Panel
    a = Scene(xlim=(0, 1), ylim=(0, 1))
    a.add(Curve(_line([(0, 0), (1, 1)])))
    b = Scene(xlim=(0, 1), ylim=(0, 1))
    b.add(Curve(_line([(0, 1), (1, 0)])))
    fig = Figure(panels=[Panel(a), Panel(b)])
    assert signals(fig, DEFAULT_STYLE, width_px=1000.0)


def test_signals_are_never_diagnostics():
    """Signals ride Report.signals, not Report.diagnostics: a sparse
    figure still PASSES."""
    from figlib.program import Report
    r = Report("t", "c", None, None, diagnostics=[],
               signals=["ink 1% — sparse"])
    assert r.passed
    assert "sparse" in r.summary()
