"""Chromatic shading ramps: light -> shadow through hue, not just lightness.

The mechanism behind riso still-life shading: a lit face drifts warm, a
shadowed face drifts cool, and the drift lives in OKLCh so lightness stays
the ordered channel (grammar: lightness is order). Gamut handling is
chroma reduction via `oklch_to_rgb`, never RGB clipping — clipping shifts
hue, the exact channel this module exists to control.
"""

from __future__ import annotations

import math
from typing import Callable

from .color import oklch_to_rgb, to_hex, to_oklab

Ramp = Callable[[float], str]


def chroma_ramp(base: str, *,
                l_range: tuple[float, float] = (0.32, 0.90),
                hue_cool: float = -45.0,
                hue_warm: float = 40.0,
                c_scale: tuple[float, float] = (1.0, 1.0)) -> Ramp:
    """t in [0, 1], shadow -> lit: hue rotates cool -> warm around `base`.

    `base` anchors hue and chroma. t=0 sits at l_range[0] with the hue
    rotated `hue_cool` degrees; t=1 at l_range[1] rotated `hue_warm`.
    Rotation sign is relative to the base hue angle in OKLab, so "warm"
    is a knob, not a promise — pick signs per base color. `c_scale`
    scales the base chroma at the two ends (shadows often want more).
    """
    _, a, b = to_oklab(base)
    c0 = math.hypot(a, b)
    h0 = math.atan2(b, a)
    l_lo, l_hi = l_range

    def ramp(t: float) -> str:
        u = min(max(float(t), 0.0), 1.0)
        L = l_lo + u * (l_hi - l_lo)
        h = h0 + math.radians(hue_cool + u * (hue_warm - hue_cool))
        C = c0 * (c_scale[0] + u * (c_scale[1] - c_scale[0]))
        rgb = oklch_to_rgb(L, C, h)
        return to_hex(tuple(float(v) for v in rgb))

    return ramp


def quantize(ramp: Ramp, bands: int) -> Ramp:
    """Posterize a ramp into exactly `bands` steps sampled at band centers."""
    if bands < 2:
        raise ValueError(f"bands must be >= 2, got {bands}")

    def q(t: float) -> str:
        u = min(max(float(t), 0.0), 1.0)
        k = min(int(u * bands), bands - 1)
        return ramp((k + 0.5) / bands)

    return q
