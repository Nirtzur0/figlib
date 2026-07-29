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

import numpy as np

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


def is_monotone(colors: list[str], min_step: float = 0.0,
                tol: float = 0.0) -> tuple[bool, str]:
    """Does this ramp read as ordered? Checks OKLab L is strictly monotone.

    Multi-hue ramps are legitimate here (the riso ramp is four hues) —
    what makes a ramp a ramp is monotone lightness, not a single hue.

    `tol` forgives reversals smaller than itself. It exists for *sampled*
    ramps: `ramp(t)` rounds to 8-bit hex, and one code step is ~0.003 in L,
    so a densely sampled ramp is full of flat and hair-width backward steps
    that are quantization, not design. Stop lists are checked at tol = 0.
    """
    ls = [lightness(c) for c in colors]
    steps = [b - a for a, b in zip(ls, ls[1:])]
    if not steps:
        return True, ""
    # Direction comes from the first step that clears the tolerance; with
    # tol = 0 that is steps[0], i.e. the original behaviour.
    sign = 0.0
    for s in steps:
        if abs(s) > tol:
            sign = 1.0 if s > 0 else -1.0
            break
    if sign == 0.0:
        sign = 1.0 if ls[-1] >= ls[0] else -1.0
    for i, (s, c0, c1) in enumerate(zip(steps, colors, colors[1:])):
        if s * sign <= -tol:
            return False, (f"L reverses at stop {i}->{i + 1} "
                           f"({c0} L={ls[i]:.3f} -> {c1} L={ls[i + 1]:.3f})")
        if abs(s) < min_step:
            return False, (f"stop {i}->{i + 1} ({c0}->{c1}) steps only "
                           f"{abs(s):.3f} in L, below {min_step:.3f}")
    return True, ""


# --- the inverse transform, and interpolation that happens inside OKLab -----
#
# The forward half above measures colors; these run it backwards, which is
# what a *ramp* needs. `ramp(t)` promises order, and order lives in L — so
# the stops have to be mixed in the space where L is a coordinate. Mixing
# '#rrggbb' channels instead hands the middle of every ramp to the sRGB
# transfer curve: the RISO ramp's sampled lightness wandered 0.010 off the
# straight path between its stops and reversed by ~0.001 inside the
# mustard->brick leg, which reads as a stall in a level-set family.

_OKLAB_TO_LMS = ((1.0, 0.3963377774, 0.2158037573),
                 (1.0, -0.1055613458, -0.0638541728),
                 (1.0, -0.0894841775, -1.2914855480))
_LMS_TO_LINEAR = ((4.0767416621, -3.3077115913, 0.2309699292),
                  (-1.2684380046, 2.6097574011, -0.3413193965),
                  (-0.0041960863, -0.7034186147, 1.7076147010))


def _to_srgb(u: float) -> float:
    """Linear light -> sRGB code value (the inverse of `_to_linear`)."""
    return 12.92 * u if u <= 0.0031308 else 1.055 * u ** (1 / 2.4) - 0.055


def to_oklab(c: str) -> tuple[float, float, float]:
    """'#rrggbb' -> OKLab (L, a, b)."""
    return _oklab(*_linear(c))


def _oklab_to_linear(L: float, a: float, b: float) -> tuple[float, float, float]:
    lms = tuple((row[0] * L + row[1] * a + row[2] * b) ** 3 for row in _OKLAB_TO_LMS)
    return tuple(sum(row[k] * lms[k] for k in range(3)) for row in _LMS_TO_LINEAR)


def from_oklab(L: float, a: float, b: float) -> str:
    """OKLab -> '#rrggbb', clipped per channel in LINEAR light.

    Clipping is the honest simple answer here: it lands an unrepresentable
    triple on the sRGB boundary instead of wrapping to a nonsense hex. It
    does shift hue and lightness, so it is only acceptable where excursions
    are small — interpolating between two in-gamut stops. The cyclic phase
    channel, which deliberately holds C constant all the way around the hue
    circle, uses `oklch_to_rgb`'s chroma reduction instead.
    """
    lin = _oklab_to_linear(L, a, b)
    return to_hex(tuple(_to_srgb(min(max(v, 0.0), 1.0)) for v in lin))


def mix_oklab(c0: str, c1: str, f: float) -> str:
    """c0 -> c1 at fraction f, interpolated in OKLab.

    L is then linear in f, so equal steps in the ramp parameter are equal
    steps in perceived lightness — the property that makes a ramp readable
    as an ordered scale rather than merely an ordered one.
    """
    f = min(max(f, 0.0), 1.0)
    a, b = to_oklab(c0), to_oklab(c1)
    return from_oklab(*[(1.0 - f) * u + f * v for u, v in zip(a, b)])


# --- the fix side of the faint-ink gate -------------------------------------
#
# The gate measures; these compute the repair, so the diagnostic can print a
# color the author types back in instead of leaving a search to the eye.


def compliant(ink_hex: str, paper_hex: str, floor: float, *,
              opacity: float = 1.0) -> str | None:
    """The nearest hue-preserving ink that clears `floor` on `paper_hex`.

    OKLab a/b are held (hue and chroma untouched up to gamut clipping) and
    only L moves — first away from the paper's lightness, then, if that
    side cannot reach the floor, toward the other extreme. Bisection finds
    the minimal move; contrast is measured on the composite at `opacity`,
    the same quantity the gate measures. None when no L clears the floor
    at this opacity.
    """
    L0, a, b = to_oklab(ink_hex)

    def clears(L: float) -> bool:
        c = composite(from_oklab(L, a, b), opacity, paper_hex)
        return contrast(c, paper_hex) >= floor

    if clears(L0):
        return ink_hex
    away = 0.0 if L0 <= lightness(paper_hex) else 1.0
    for extreme in (away, 1.0 - away):
        if not clears(extreme):
            continue
        lo, hi = L0, extreme            # lo fails, hi clears
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            if clears(mid):
                hi = mid
            else:
                lo = mid
        return from_oklab(hi, a, b)
    return None


def min_compliant_opacity(ink_hex: str, paper_hex: str,
                          floor: float) -> float | None:
    """The smallest opacity at which `ink_hex` composited on `paper_hex`
    clears `floor` — rounded UP to 0.01 so the printed value still clears.
    None when even opacity 1.0 fails (the hex itself is the problem)."""
    def clears(alpha: float) -> bool:
        return contrast(composite(ink_hex, alpha, paper_hex), paper_hex) >= floor

    if not clears(1.0):
        return None
    lo, hi = 0.0, 1.0                   # lo fails (alpha 0 = paper), hi clears
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        if clears(mid):
            hi = mid
        else:
            lo = mid
    up = math.ceil(hi * 100.0) / 100.0
    return up if clears(up) else min(up + 0.01, 1.0)


# --- OKLCh, vectorized: the cyclic (phase) channel --------------------------

_LAB_TO_LMS_M = np.array(_OKLAB_TO_LMS)
_LMS_TO_LIN_M = np.array(_LMS_TO_LINEAR)
_GAMUT_EPS = 1e-7


def _oklab_to_linear_arr(lab: np.ndarray) -> np.ndarray:
    """(..., 3) OKLab -> (..., 3) linear sRGB, unclipped."""
    lms = lab @ _LAB_TO_LMS_M.T
    return (lms ** 3) @ _LMS_TO_LIN_M.T


def _srgb_arr(lin: np.ndarray) -> np.ndarray:
    u = np.clip(lin, 0.0, 1.0)
    return np.where(u <= 0.0031308, 12.92 * u, 1.055 * u ** (1 / 2.4) - 0.055)


def oklch_to_rgb(lightness_: np.ndarray | float, chroma_: np.ndarray | float,
                 hue_rad: np.ndarray | float, *, iters: int = 14) -> np.ndarray:
    """Vectorized OKLCh -> sRGB in [0, 1], shape (..., 3).

    Gamut handling is chroma reduction, not per-channel clipping: hue and
    lightness are held exactly and C is bisected down to the largest value
    that is representable. That is the right trade for a cyclic channel,
    where the two things carrying meaning are the *evenness of the hue
    circle* (identity of the phase) and *lightness* (the modulus), while
    chroma carries nothing. Clipping would corrupt both of the former to
    preserve the latter, which nothing asked for.

    Cost is `iters` vectorized passes over the array; at 400x400 that is
    milliseconds, so the per-pixel path can afford the exact answer.
    """
    L = np.clip(np.broadcast_arrays(np.asarray(lightness_, dtype=float),
                                    np.asarray(hue_rad, dtype=float))[0], 0.0, 1.0)
    C = np.broadcast_to(np.asarray(chroma_, dtype=float), L.shape)
    h = np.broadcast_to(np.asarray(hue_rad, dtype=float), L.shape)
    ca, cb = np.cos(h), np.sin(h)

    def fits(scale: np.ndarray) -> np.ndarray:
        lab = np.stack([L, scale * C * ca, scale * C * cb], axis=-1)
        lin = _oklab_to_linear_arr(lab)
        return ((lin >= -_GAMUT_EPS) & (lin <= 1.0 + _GAMUT_EPS)).all(axis=-1)

    full = fits(np.ones_like(L))
    lo = np.zeros_like(L)
    hi = np.ones_like(L)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        ok = fits(mid)
        lo = np.where(ok, mid, lo)
        hi = np.where(ok, hi, mid)
    s = np.where(full, 1.0, lo)
    lab = np.stack([L, s * C * ca, s * C * cb], axis=-1)
    return _srgb_arr(_oklab_to_linear_arr(lab))


def rgb_to_hex_arr(rgb: np.ndarray) -> np.ndarray:
    """(..., 3) float sRGB in [0, 1] -> (...,) array of '#rrggbb' strings.
    For measuring a vector channel with the scalar colorimetry above."""
    b = np.round(np.clip(rgb, 0.0, 1.0) * 255).astype(int)
    flat = b.reshape(-1, 3)
    hexes = np.array(["#%02x%02x%02x" % tuple(v) for v in flat])
    return hexes.reshape(rgb.shape[:-1])
