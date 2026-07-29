"""chroma_ramp: lightness stays the ordered channel while hue drifts."""

import math

import numpy as np
import pytest

from figlib.color import is_monotone, lightness, to_oklab
from figlib.render import to_svg
from figlib.scene import FilledCurve, Gradient, Scene
from figlib.shading import chroma_ramp, quantize
from figlib.style import DEFAULT_STYLE

_TRI = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])


def _svg_of(items, style=DEFAULT_STYLE):
    s = Scene(items=list(items), xlim=(-0.2, 1.2), ylim=(-0.2, 1.2))
    return to_svg(s, style, width_px=300)


def test_ramp_lightness_monotone():
    ramp = chroma_ramp("#c0504d")
    samples = [ramp(k / 40) for k in range(41)]
    ok, why = is_monotone(samples, tol=0.004)
    assert ok, why


def test_ramp_endpoints_hit_l_range():
    ramp = chroma_ramp("#c0504d", l_range=(0.30, 0.90))
    assert abs(lightness(ramp(0.0)) - 0.30) < 0.03
    assert abs(lightness(ramp(1.0)) - 0.90) < 0.03


def test_ramp_hue_actually_drifts():
    # cool end and warm end must differ in hue, not only lightness
    ramp = chroma_ramp("#c0504d", hue_cool=-45.0, hue_warm=40.0)

    def hue(c):
        _, a, b = to_oklab(c)
        return math.atan2(b, a)

    dh = (hue(ramp(1.0)) - hue(ramp(0.0))) % (2 * math.pi)
    if dh > math.pi:
        dh -= 2 * math.pi
    assert abs(math.degrees(dh)) > 30.0


def test_ramp_colors_are_shade_tagged():
    # the color gate exempts SHADE-channel fills; the ramp's own
    # readability contract is the monotone-lightness test above
    from figlib.theme import SHADE
    ramp = chroma_ramp("#c0504d")
    assert getattr(ramp(0.5), "channel", None) == SHADE


def test_ramp_clamps_t():
    ramp = chroma_ramp("#3a6ea5")
    assert ramp(-1.0) == ramp(0.0)
    assert ramp(2.0) == ramp(1.0)


def test_quantize_band_count():
    ramp = chroma_ramp("#3a6ea5")
    q = quantize(ramp, 4)
    colors = {q(k / 100) for k in range(101)}
    assert len(colors) == 4


def test_quantize_rejects_degenerate():
    with pytest.raises(ValueError):
        quantize(chroma_ramp("#3a6ea5"), 1)


def test_gradient_from_ramp_stops():
    ramp = chroma_ramp("#3a6ea5")
    g = Gradient.from_ramp(ramp, (0.0, 0.0), (1.0, 0.0), n=5)
    assert len(g.stops) == 5
    assert g.stops[0][0] == 0.0 and g.stops[-1][0] == 1.0
    assert g.stops[0][1] == ramp(0.0) and g.stops[-1][1] == ramp(1.0)
    assert g.kind == "linear"


def test_gradient_from_ramp_t_range():
    ramp = chroma_ramp("#3a6ea5")
    g = Gradient.from_ramp(ramp, (0.0, 0.0), (1.0, 0.0), t_range=(0.4, 0.6), n=3)
    assert g.stops[1][1] == ramp(0.5)


def test_filledcurve_gradient_and_pattern_rejected():
    ramp = chroma_ramp("#3a6ea5")
    g = Gradient.from_ramp(ramp, (0.0, 0.0), (1.0, 0.0))
    with pytest.raises(ValueError):
        FilledCurve(_TRI, gradient=g, pattern="stipple")


def test_filledcurve_gradient_alone_ok():
    ramp = chroma_ramp("#3a6ea5")
    g = Gradient.from_ramp(ramp, (0.0, 0.0), (1.0, 0.0))
    fc = FilledCurve(_TRI, gradient=g, grain=0.3, opacity=1.0)
    assert fc.gradient is g and fc.grain == 0.3


def test_gradient_def_emitted_userspace():
    ramp = chroma_ramp("#3a6ea5")
    g = Gradient.from_ramp(ramp, (0.0, 0.0), (1.0, 0.0))
    svg = _svg_of([FilledCurve(_TRI, gradient=g, opacity=1.0)])
    assert 'gradientUnits="userSpaceOnUse"' in svg
    assert "linearGradient" in svg
    assert 'fill="url(#grad-' in svg


def test_gradient_defs_deduped():
    ramp = chroma_ramp("#3a6ea5")
    g = Gradient.from_ramp(ramp, (0.0, 0.0), (1.0, 0.0))
    svg = _svg_of([FilledCurve(_TRI, gradient=g, opacity=1.0),
                   FilledCurve(_TRI + 0.05, gradient=g, opacity=1.0)])
    assert svg.count("<linearGradient") == 1


def test_radial_gradient_emitted():
    ramp = chroma_ramp("#3a6ea5")
    g = Gradient.from_ramp(ramp, (0.5, 0.5), (1.0, 0.5), kind="radial")
    svg = _svg_of([FilledCurve(_TRI, gradient=g, opacity=1.0)])
    assert "radialGradient" in svg


def test_fill_grain_overlay_emitted():
    svg = _svg_of([FilledCurve(_TRI, color="#c0504d", opacity=1.0, grain=0.4)])
    assert 'fill="url(#grain)"' in svg


def test_fill_grain_skipped_when_zero():
    svg = _svg_of([FilledCurve(_TRI, color="#c0504d", opacity=1.0)])
    assert 'fill="url(#grain)"' not in svg


def test_fill_grain_skipped_on_transparent_theme():
    from figlib.theme import RISO_T
    svg = _svg_of([FilledCurve(_TRI, color="#c0504d", opacity=1.0, grain=0.4)],
                  style=RISO_T)
    assert 'fill="url(#grain)"' not in svg


def test_expressivity_counts_gradient_channel():
    from figlib.expressivity import signals
    ramp = chroma_ramp("#3a6ea5")
    g = Gradient.from_ramp(ramp, (0.0, 0.0), (1.0, 0.0))
    s = Scene(items=[FilledCurve(_TRI, gradient=g, opacity=1.0, grain=0.2)],
              xlim=(-0.2, 1.2), ylim=(-0.2, 1.2))
    chan_line = next(ln for ln in signals(s, DEFAULT_STYLE, width_px=300.0)
                     if ln.startswith("channels:"))
    used = chan_line.split(";")[0]
    assert "gradient" in used and "grain" in used
