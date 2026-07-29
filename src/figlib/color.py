"""Colorimetry: the measurements that make the color channel gateable.

grammar.md asserts three properties of color — hue is correspondence,
lightness is order, and ink must be visible on paper. All three are
computable, so none of them are matters of taste:

    contrast(a, b)        WCAG ratio -> is this stroke visible on paper?
    delta_e(a, b, cvd)    OKLab distance -> are two hues distinguishable?
    lightness(c)          OKLab L -> does a ramp actually read as ordered?
    composite(c, alpha, paper)   what the reader sees, not what we asked for

Deficiency simulation is Machado-Oliveira-Fernandes (2009) at severity
1.0; the thresholds below are calibrated against that model, so the model
is part of the standard rather than an implementation detail.
"""

from __future__ import annotations

import math

# Machado-Oliveira-Fernandes 2009, severity 1.0, linear-RGB.
_MACHADO = {
    "protan": ((0.152286, 1.052583, -0.204868),
               (0.114503, 0.786281, 0.099216),
               (-0.003882, -0.048116, 1.051998)),
    "deutan": ((0.367322, 0.860646, -0.227968),
               (0.280085, 0.672501, 0.047413),
               (-0.011820, 0.042940, 0.968881)),
    "tritan": ((1.255528, -0.076749, -0.178779),
               (-0.078411, 0.930809, 0.147602),
               (0.004733, 0.691367, 0.303900)),
}

CVD_KINDS = ("protan", "deutan", "tritan")
# Gated conditions. Tritan is ~0.01% prevalence against ~8% for the red-green
# pair, so it is measured and reported but does not fail a palette.
GATE_KINDS = ("protan", "deutan")


def to_rgb(c: str) -> tuple[float, float, float]:
    """'#rrggbb' (or a bare 'white'/'black') -> sRGB in [0, 1]."""
    if not c.startswith("#"):
        named = {"white": "#ffffff", "black": "#000000", "none": "#ffffff"}
        c = named.get(c.lower(), c)
    h = c.lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def to_hex(rgb: tuple[float, float, float]) -> str:
    return "#" + "".join(f"{round(min(max(v, 0.0), 1.0) * 255):02x}" for v in rgb)


def _to_linear(u: float) -> float:
    return u / 12.92 if u <= 0.04045 else ((u + 0.055) / 1.055) ** 2.4


def _linear(c: str) -> tuple[float, float, float]:
    return tuple(_to_linear(v) for v in to_rgb(c))


def _oklab(r: float, g: float, b: float) -> tuple[float, float, float]:
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l, m, s = l ** (1 / 3), m ** (1 / 3), s ** (1 / 3)
    return (0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s)


def lightness(c: str) -> float:
    """OKLab L in [0, 1] — perceptual lightness, the 'order' channel."""
    return _oklab(*_linear(c))[0]


def chroma(c: str) -> float:
    """OKLab chroma — below ~0.10 a hue stops doing identity work."""
    _, a, b = _oklab(*_linear(c))
    return math.hypot(a, b)


def relative_luminance(c: str) -> float:
    r, g, b = _linear(c)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    """WCAG contrast ratio, 1.0 (identical) to 21.0 (black on white)."""
    la, lb = relative_luminance(a), relative_luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def simulate(c: str, kind: str) -> str:
    """c as seen under protan / deutan / tritan deficiency."""
    r, g, b = _linear(c)
    m = _MACHADO[kind]
    lin = tuple(row[0] * r + row[1] * g + row[2] * b for row in m)

    def encode(u: float) -> float:
        u = min(max(u, 0.0), 1.0)
        return 12.92 * u if u <= 0.0031308 else 1.055 * u ** (1 / 2.4) - 0.055

    return to_hex(tuple(encode(v) for v in lin))


def delta_e(a: str, b: str, cvd: str | None = None) -> float:
    """Euclidean OKLab distance x100. cvd=None is normal vision."""
    pa = _oklab(*_linear(simulate(a, cvd) if cvd else a))
    pb = _oklab(*_linear(simulate(b, cvd) if cvd else b))
    return 100 * math.dist(pa, pb)


def worst_delta_e(a: str, b: str,
                  kinds: tuple[str, ...] = GATE_KINDS) -> tuple[float, str]:
    """The (distance, condition) under which this pair is hardest to tell
    apart — normal vision included, since a pair can collapse there too."""
    trials = [(delta_e(a, b), "normal")]
    trials += [(delta_e(a, b, k), k) for k in kinds]
    return min(trials)


def composite(c: str, alpha: float, paper: str) -> str:
    """What the reader actually sees: c drawn at `alpha` over `paper`.

    Measuring the declared color of a 0.3-opacity stroke overstates its
    visibility by a factor of three; measure the composite instead.
    """
    if alpha >= 1.0:
        return c
    fg, bg = to_rgb(c), to_rgb(paper)
    return to_hex(tuple(alpha * f + (1 - alpha) * b for f, b in zip(fg, bg)))


def is_monotone(colors: list[str], min_step: float = 0.0) -> tuple[bool, str]:
    """Does this ramp read as ordered? Checks OKLab L is strictly monotone.

    Multi-hue ramps are legitimate here (the riso ramp is four hues) —
    what makes a ramp a ramp is monotone lightness, not a single hue.
    """
    ls = [lightness(c) for c in colors]
    steps = [b - a for a, b in zip(ls, ls[1:])]
    if not steps:
        return True, ""
    sign = 1.0 if steps[0] > 0 else -1.0
    for i, (s, c0, c1) in enumerate(zip(steps, colors, colors[1:])):
        if s * sign <= 0:
            return False, (f"L reverses at stop {i}->{i + 1} "
                           f"({c0} L={ls[i]:.3f} -> {c1} L={ls[i + 1]:.3f})")
        if abs(s) < min_step:
            return False, (f"stop {i}->{i + 1} ({c0}->{c1}) steps only "
                           f"{abs(s):.3f} in L, below {min_step:.3f}")
    return True, ""
