# Chromatic Gradient Shading + 3D Solids Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Gradient fills as a first-class scene channel, OKLCh chromatic shading ramps, fill-scoped grain, and extruded 3D solids (box/cylinder/prism), proven by a reference-style demo figure.

**Architecture:** A `Gradient` spec lives in `scene.py` (math coords, like every primitive); `render.py` emits deduped SVG gradient defs through the layout `Transform`. A new `shading.py` builds hue-shifted ramps in OKLCh on top of `color.py`'s existing OKLab machinery. A new `solids.py` turns extruded prisms into depth-tagged `FilledCurve` faces, each carrying a light-aligned gradient — it composes with the existing `surface3d` painter's-algorithm pipeline unchanged.

**Tech Stack:** Python 3.11+, numpy, existing figlib modules (`color`, `scene`, `render`, `surface3d`, `expressivity`). No new dependencies.

Spec: `docs/superpowers/specs/2026-07-29-gradient-shading-design.md`

## Global Constraints

- Gamut mapping in ramps is chroma reduction (`color.oklch_to_rgb`), never per-channel RGB clipping.
- Ramp lightness must be monotone in t (`color.is_monotone` with sampling tolerance).
- `Gradient` coordinates are MATH coords; only `layout.Transform` converts to canvas px (project rule).
- `gradient` + `pattern` on one `FilledCurve` is a `ValueError`.
- Fill grain reuses the tile-pattern route (`theme.grain_tile_datauri`), never `feTurbulence`.
- Run tests from the repo root: `python -m pytest tests/<file> -v` (package lives in `src/`, installed editable).
- Solids deviate from the spec's "in surface3d.py" placement: they go in a NEW `src/figlib/solids.py` (surface3d stays projection + heightfields; solids are polyhedra). Everything else per spec.

---

### Task 1: Chromatic ramps — `shading.py`

**Files:**
- Create: `src/figlib/shading.py`
- Test: `tests/test_shading.py`

**Interfaces:**
- Consumes: `color.to_oklab(hex) -> (L, a, b)`, `color.oklch_to_rgb(L, C, h_rad) -> ndarray(...,3)`, `color.to_hex(rgb_tuple)`, `color.is_monotone(list[str], tol=...)`.
- Produces: `Ramp = Callable[[float], str]`; `chroma_ramp(base, *, l_range, hue_cool, hue_warm, c_scale) -> Ramp`; `quantize(ramp, bands) -> Ramp`. Tasks 6–7 pass these as the `ramp` argument.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_shading.py
"""chroma_ramp: lightness stays the ordered channel while hue drifts."""

import pytest

from figlib.color import is_monotone, lightness, to_oklab
from figlib.shading import chroma_ramp, quantize


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
    import math
    ramp = chroma_ramp("#c0504d", hue_cool=-45.0, hue_warm=40.0)
    def hue(c):
        _, a, b = to_oklab(c)
        return math.atan2(b, a)
    dh = (hue(ramp(1.0)) - hue(ramp(0.0))) % (2 * math.pi)
    if dh > math.pi:
        dh -= 2 * math.pi
    assert abs(math.degrees(dh)) > 30.0


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_shading.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'figlib.shading'`

- [ ] **Step 3: Write the implementation**

```python
# src/figlib/shading.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_shading.py -v`
Expected: 6 PASS

- [ ] **Step 5: Commit**

```bash
git add src/figlib/shading.py tests/test_shading.py
git commit -m "shading: OKLCh chromatic ramps (cool shadow / warm light) + quantize"
```

---

### Task 2: `Gradient` primitive + `FilledCurve.gradient`/`.grain` — `scene.py`

**Files:**
- Modify: `src/figlib/scene.py` (Gradient before FilledCurve; FilledCurve gains two fields + `__post_init__`)
- Test: `tests/test_shading.py` (append)

**Interfaces:**
- Consumes: `shading.Ramp` (Task 1) in `Gradient.from_ramp`.
- Produces: `Gradient(stops, kind="linear", p0, p1)` frozen dataclass; `Gradient.from_ramp(ramp, p0, p1, *, t_range=(0.0, 1.0), n=5, kind="linear") -> Gradient`; `FilledCurve.gradient: Gradient | None`, `FilledCurve.grain: float`. Tasks 3–6 rely on these exact names.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_shading.py`)

```python
import numpy as np

from figlib.scene import FilledCurve, Gradient

_TRI = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_shading.py -v`
Expected: new tests FAIL with `ImportError: cannot import name 'Gradient'`

- [ ] **Step 3: Implement**

In `src/figlib/scene.py`, immediately BEFORE the `FilledCurve` dataclass, add:

```python
@dataclass(frozen=True)
class Gradient:
    """A fill paint: linear or radial color ramp, axis in MATH coords.

    p0 -> p1 is the linear axis (radial: center -> a point on the rim).
    Coordinates are scene coords like every primitive; only
    layout.Transform converts them at emission.
    """
    stops: tuple[tuple[float, str], ...]   # (offset in [0,1], "#rrggbb")
    kind: str = "linear"                   # "linear" | "radial"
    p0: XY = (0.0, 0.0)
    p1: XY = (1.0, 0.0)

    @staticmethod
    def from_ramp(ramp: Callable[[float], str], p0: XY, p1: XY, *,
                  t_range: tuple[float, float] = (0.0, 1.0), n: int = 5,
                  kind: str = "linear") -> "Gradient":
        """Sample `ramp` over t_range into n evenly spaced stops."""
        t0, t1 = t_range
        stops = tuple((k / (n - 1), ramp(t0 + (t1 - t0) * k / (n - 1)))
                      for k in range(n))
        return Gradient(stops=stops, kind=kind,
                        p0=(float(p0[0]), float(p0[1])),
                        p1=(float(p1[0]), float(p1[1])))
```

In `FilledCurve`, after the `holes` field, add:

```python
    # gradient fill paint (overrides color); axis in math coords
    gradient: Gradient | None = None
    # grain INSIDE the fill: 0 = none, else opacity of the grain tile
    # overlaid clipped to this shape (skipped on transparent themes)
    grain: float = 0.0

    def __post_init__(self) -> None:
        if self.gradient is not None and self.pattern is not None:
            raise ValueError(
                "FilledCurve: gradient and pattern are competing fill paints")
```

(`Callable` is already imported in scene.py; verify, else add to the typing import.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_shading.py tests/test_core.py -v`
Expected: PASS (test_core guards against dataclass regressions)

- [ ] **Step 5: Commit**

```bash
git add src/figlib/scene.py tests/test_shading.py
git commit -m "scene: Gradient fill primitive; FilledCurve gradient + fill-scoped grain fields"
```

---

### Task 3: Gradient emission — `render.py`

**Files:**
- Modify: `src/figlib/render.py` (new `_ensure_gradient`; wire into the FilledCurve branch of `_emit_items`, currently around lines 417–440)
- Test: `tests/test_shading.py` (append)

**Interfaces:**
- Consumes: `scene.Gradient` (Task 2), `layout.Transform.to_canvas`.
- Produces: `_ensure_gradient(defs, grad, t) -> str` (def id); SVG output where a gradient-filled path has `fill="url(#grad-…)"` and defs contain `linearGradient`/`radialGradient` with `gradientUnits="userSpaceOnUse"`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_shading.py`)

Existing tests render scenes via helpers in `tests/svgkit.py` — open it first and reuse its render-to-SVG helper if one exists; otherwise use this pattern (mirrors test_ink_channels.py — check that file's imports for the exact scene-render entry):

```python
from figlib.render import render_svg          # adjust to the actual API in render.py
from figlib.scene import Scene
from figlib.theme import RISO


def _svg_of(items):
    s = Scene()
    s.items.extend(items)
    s.xlim = (-0.2, 1.2)
    s.ylim = (-0.2, 1.2)
    return render_svg(s, RISO.style(), width_px=300)   # adjust to actual signature


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
```

NOTE to implementer: the exact render entry point must be taken from `render.py` / existing tests (e.g. `tests/test_ink_channels.py`) — fix `_svg_of` accordingly before writing the implementation. The assertions are the contract; the helper is plumbing.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_shading.py -v -k gradient`
Expected: the three new tests FAIL (no `url(#grad-` in output)

- [ ] **Step 3: Implement**

In `render.py`, next to `_ensure_pattern` (~line 260), add (`hashlib` to imports; `math` is already imported):

```python
def _ensure_gradient(defs: ET.Element, grad: "Gradient", t: Transform) -> str:
    """One gradient def per (kind, stops, canvas axis), content-addressed."""
    x0, y0 = t.to_canvas(grad.p0)
    x1, y1 = t.to_canvas(grad.p1)
    key = (grad.kind, grad.stops, round(x0, 2), round(y0, 2),
           round(x1, 2), round(y1, 2))
    gid = "grad-" + hashlib.md5(repr(key).encode()).hexdigest()[:10]
    tag = "linearGradient" if grad.kind == "linear" else "radialGradient"
    if defs.find(f"./{tag}[@id='{gid}']") is not None:
        return gid
    if grad.kind == "linear":
        el = ET.SubElement(defs, "linearGradient", {
            "id": gid, "gradientUnits": "userSpaceOnUse",
            "x1": _fmt(x0), "y1": _fmt(y0), "x2": _fmt(x1), "y2": _fmt(y1)})
    elif grad.kind == "radial":
        el = ET.SubElement(defs, "radialGradient", {
            "id": gid, "gradientUnits": "userSpaceOnUse",
            "cx": _fmt(x0), "cy": _fmt(y0),
            "r": _fmt(math.hypot(x1 - x0, y1 - y0))})
    else:
        raise ValueError(f"unknown gradient kind {grad.kind!r}")
    for off, color in grad.stops:
        ET.SubElement(el, "stop", {"offset": _fmt(off), "stop-color": color})
    return gid
```

In the FilledCurve branch of `_emit_items`, extend the fill-paint choice (currently `if it.pattern is not None and defs is not None: ... else: ...`):

```python
            if it.gradient is not None and defs is not None:
                attrs["fill"] = f"url(#{_ensure_gradient(defs, it.gradient, t)})"
                if it.opacity < 1.0:
                    attrs["fill-opacity"] = _fmt(it.opacity)
            elif it.pattern is not None and defs is not None:
                # texture is ink, not a wash — item opacity does not apply
                attrs["fill"] = f"url(#{_ensure_pattern(defs, it.pattern, fill_color)})"
            else:
                attrs["fill"] = fill_color
                attrs["fill-opacity"] = _fmt(it.opacity)
```

Also add `Gradient` to the `from .scene import (...)` in render.py.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_shading.py tests/test_ink_channels.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/figlib/render.py tests/test_shading.py
git commit -m "render: SVG linear/radial gradient defs, userSpaceOnUse, content-deduped"
```

---

### Task 4: Fill-scoped grain — `render.py`

**Files:**
- Modify: `src/figlib/render.py` (`_ensure_grain_pattern` factored out of `_emit_grain` ~line 150; grain overlay path in the FilledCurve branch)
- Test: `tests/test_shading.py` (append)

**Interfaces:**
- Consumes: `FilledCurve.grain` (Task 2), `theme.grain_tile_datauri` (existing).
- Produces: `_ensure_grain_pattern(defs) -> str` (returns `"grain"`), used by both the page overlay and fill overlays.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_shading.py`)

```python
def test_fill_grain_overlay_emitted():
    svg = _svg_of([FilledCurve(_TRI, color="#c0504d", opacity=1.0, grain=0.4)])
    assert 'fill="url(#grain)"' in svg


def test_fill_grain_skipped_when_zero():
    svg = _svg_of([FilledCurve(_TRI, color="#c0504d", opacity=1.0)])
    # the page-level grain rect may exist; no per-fill overlay path
    assert svg.count('fill="url(#grain)"') <= 1
```

The transparent-theme skip: render the grain item with the transparent style (see how `tests/` exercise `transparent` — `run(..., transparent=True)` in cli or a Style flag) and assert no `url(#grain)` appears at all. If no cheap hook exists in the test harness, assert on the code path guard instead (transparent check mirrors the existing `casing` guard) and leave the rendered check to the demo task.

```python
def test_fill_grain_skipped_on_transparent_theme():
    import dataclasses
    style = dataclasses.replace(RISO.style(), transparent=True)   # adjust to actual Style API
    s = Scene(); s.items.append(FilledCurve(_TRI, color="#c0504d", opacity=1.0, grain=0.4))
    s.xlim = (-0.2, 1.2); s.ylim = (-0.2, 1.2)
    svg = render_svg(s, style, width_px=300)
    assert 'fill="url(#grain)"' not in svg
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_shading.py -v -k grain`
Expected: `test_fill_grain_overlay_emitted` FAILS

- [ ] **Step 3: Implement**

Factor the pattern def out of `_emit_grain` (which currently inlines it):

```python
def _ensure_grain_pattern(defs: ET.Element) -> str:
    """The shared grain tile pattern def (page overlay + fill overlays)."""
    if defs.find("./pattern[@id='grain']") is None:
        from .theme import grain_tile_datauri
        pat = ET.SubElement(defs, "pattern", {
            "id": "grain", "patternUnits": "userSpaceOnUse",
            "width": "140", "height": "140"})
        ET.SubElement(pat, "image", {
            "href": grain_tile_datauri(), "width": "140", "height": "140"})
    return "grain"
```

Rewrite `_emit_grain`'s def-creation lines to call `_ensure_grain_pattern(defs)` (the page `<rect>` emission stays as is).

In the FilledCurve branch, AFTER the main path element `el` (and its stroke attributes) — so grain sits on top of the fill but under later items:

```python
            if it.grain > 0 and not transparent and defs is not None:
                gattrs = {"d": d,
                          "fill": f"url(#{_ensure_grain_pattern(defs)})",
                          "fill-opacity": _fmt(it.grain), "stroke": "none"}
                if it.holes:
                    gattrs["fill-rule"] = "evenodd"
                ET.SubElement(root, "path", gattrs)
```

(Same `d` string as the fill path — identical geometry needs no clipPath.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_shading.py -v`
Expected: PASS. Also run `python -m pytest tests/ -x -q` — the `_emit_grain` refactor must not break the full suite.

- [ ] **Step 5: Commit**

```bash
git add src/figlib/render.py tests/test_shading.py
git commit -m "render: fill-scoped grain overlay reusing the shared grain tile"
```

---

### Task 5: Expressivity — gradient/grain are channels

**Files:**
- Modify: `src/figlib/expressivity.py` (FilledCurve branch ~line 123; `_NOTABLE` ~line 165)
- Test: `tests/test_shading.py` (append)

**Interfaces:**
- Consumes: `FilledCurve.gradient/.grain` (Task 2).
- Produces: `"gradient"` and `"grain"` in the survey's `channels` set; `"gradient"` counted notable.

- [ ] **Step 1: Write the failing test** (append; open `tests/test_expressivity.py` first and mirror how it constructs the survey object — reuse its helper/fixture; the assertion is the contract)

```python
def test_expressivity_counts_gradient_channel():
    # mirror tests/test_expressivity.py's survey construction
    from tests.test_expressivity import ...  # or copy its local helper
    ramp = chroma_ramp("#3a6ea5")
    g = Gradient.from_ramp(ramp, (0.0, 0.0), (1.0, 0.0))
    survey = _survey_of([FilledCurve(_TRI, gradient=g, opacity=1.0, grain=0.2)])
    assert "gradient" in survey.channels
    assert "grain" in survey.channels
```

- [ ] **Step 2: Run to verify it fails** — `python -m pytest tests/test_shading.py -v -k expressivity` → FAIL

- [ ] **Step 3: Implement** — in the `isinstance(it, FilledCurve)` branch of expressivity.py, after `self.channels.add("fill")`:

```python
            if it.gradient is not None:
                self.channels.add("gradient")
                mid = it.gradient.stops[len(it.gradient.stops) // 2][1]
                self._note_hue(mid)
            if it.grain > 0:
                self.channels.add("grain")
```

And extend `_NOTABLE` to include `"gradient"`:

```python
_NOTABLE = ("raster", "fill", "wash", "width_profile", "pattern", "casing",
            "dash", "arrows", "gradient")
```

- [ ] **Step 4: Verify** — `python -m pytest tests/test_shading.py tests/test_expressivity.py -v` → PASS

- [ ] **Step 5: Commit**

```bash
git add src/figlib/expressivity.py tests/test_shading.py
git commit -m "expressivity: gradient and grain register as channels"
```

---

### Task 6: Solids — `solids.py`

**Files:**
- Create: `src/figlib/solids.py`
- Test: `tests/test_solids.py`

**Interfaces:**
- Consumes: `surface3d.Camera/LIGHT_DIR/project`, `scene.FilledCurve/Gradient`, `shading.quantize`, `shading.Ramp` (a `t -> hex` callable).
- Produces:
  - `face_item(poly3, cam, ramp, *, grad_amp=0.12, grain=0.0, edge=None, edge_width=0.35, stops=5) -> tuple[float, FilledCurve] | None`
  - `extrude_items(poly2, z0, z1, cam, ramp, *, side_grad_amp=0.12, cap_grad_amp=0.12, bands=None, grain=0.0, edge=None, edge_width=0.35) -> list[tuple[float, FilledCurve]]`
  - `box_items(center, size, cam, ramp, **kw) -> list[tuple[float, FilledCurve]]`
  - `cylinder_items(center, radius, height, cam, ramp, *, facets=48, **kw) -> list[tuple[float, FilledCurve]]`
  - All lists compose with `surface3d.compose` / `drop_shadow` / `as_floor` unchanged.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_solids.py
"""Solids: backface culling, winding, gradients, banding."""

import numpy as np

from figlib.scene import FilledCurve
from figlib.shading import chroma_ramp
from figlib.solids import box_items, cylinder_items, extrude_items, face_item
from figlib.surface3d import Camera

CAM = Camera()                       # azim -35, elev 32
RAMP = chroma_ramp("#c0504d")


def test_face_item_backface_culled():
    # +z-facing square is visible from the default camera (elev > 0)
    top = np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)
    assert face_item(top, CAM, RAMP) is not None
    # reversed winding -> -z normal -> backface
    assert face_item(top[::-1], CAM, RAMP) is None


def test_face_item_carries_gradient():
    top = np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)
    depth, fc = face_item(top, CAM, RAMP)
    assert isinstance(fc, FilledCurve)
    assert fc.gradient is not None
    assert fc.opacity == 1.0
    # grad_amp=0 -> flat fill, no gradient
    _, flat = face_item(top, CAM, RAMP, grad_amp=0.0)
    assert flat.gradient is None and flat.color == RAMP(_lambert_of(top))


def _lambert_of(poly3):
    from figlib.solids import _lambert
    n = np.cross(poly3[1] - poly3[0], poly3[2] - poly3[0])
    return _lambert(n / np.linalg.norm(n))


def test_box_visible_faces():
    items = box_items((0, 0, 0), (1, 1, 1), CAM, RAMP)
    # a box shows exactly 3 faces from a generic camera
    assert len(items) == 3
    depths = [d for d, _ in items]
    assert all(isinstance(d, float) for d in depths)


def test_box_faces_differ_in_tone():
    items = box_items((0, 0, 0), (1, 1, 1), CAM, RAMP, side_grad_amp=0.0,
                      cap_grad_amp=0.0)
    colors = {fc.color for _, fc in items}
    assert len(colors) == 3          # three faces, three Lambert values


def test_cylinder_banding():
    items = cylinder_items((0, 0, 0), 1.0, 2.0, CAM, RAMP, facets=64, bands=4)
    sides = [fc for _, fc in items if len(fc.pts) == 4]
    caps = [fc for _, fc in items if len(fc.pts) > 4]
    assert len(caps) == 1            # top cap visible, bottom culled
    side_colors = {fc.color for fc in sides}
    assert 2 <= len(side_colors) <= 4


def test_extrude_nonconvex_runs():
    # L-shaped CCW polygon: painter's algorithm input stays well-formed
    L = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], dtype=float)
    items = extrude_items(L, 0.0, 1.0, CAM, RAMP)
    assert len(items) >= 4
    for d, fc in items:
        assert fc.pts.shape[1] == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_solids.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'figlib.solids'`

- [ ] **Step 3: Write the implementation**

```python
# src/figlib/solids.py
"""Polyhedral solids: extruded prisms, boxes, cylinders as shaded faces.

Each visible face becomes ONE FilledCurve whose fill is a linear Gradient
aligned with the light direction projected into the face plane. A flat
face under directional light has CONSTANT Lambert shade — the within-face
drift is stylistic, not physical. So the face's true Lambert value sets
its base tone and the gradient drifts +-grad_amp around it: physics sets
the tone, the gradient adds the glow.

Output is depth-tagged (depth, FilledCurve) pairs — the same contract as
surface3d.surface_items, so compose/drop_shadow/as_floor apply unchanged.
"""

from __future__ import annotations

import numpy as np

from .scene import FilledCurve, Gradient
from .shading import Ramp, quantize
from .style import Role
from .surface3d import LIGHT_DIR, Camera, project


def _lambert(n_hat: np.ndarray) -> float:
    # same ambient floor + diffuse scale as surface3d.surface_items
    return 0.25 + 0.75 * max(0.0, float(n_hat @ LIGHT_DIR))


def face_item(poly3: np.ndarray, cam: Camera, ramp: Ramp, *,
              grad_amp: float = 0.12, grain: float = 0.0,
              edge: str | None = None, edge_width: float = 0.35,
              stops: int = 5) -> tuple[float, FilledCurve] | None:
    """One planar face -> (mean depth, FilledCurve); None when backfacing.

    Winding defines the outward normal (right-hand rule); faces whose
    normal points away from the camera are culled, which is the whole
    hidden-surface story for a convex solid.
    """
    poly3 = np.asarray(poly3, dtype=float)
    n = np.cross(poly3[1] - poly3[0], poly3[2] - poly3[0])
    nn = float(np.linalg.norm(n))
    if nn == 0.0:
        return None
    n_hat = n / nn
    _, _, toward = cam.axes()
    if float(n_hat @ toward) <= 1e-9:
        return None
    t = _lambert(n_hat)
    pts2, depth = project(poly3, cam)

    grad = None
    if grad_amp > 0.0:
        g3 = LIGHT_DIR - float(LIGHT_DIR @ n_hat) * n_hat
        gn = float(np.linalg.norm(g3))
        if gn > 1e-6:
            g_hat = g3 / gn
            c = poly3.mean(axis=0)
            s = float(np.abs((poly3 - c) @ g_hat).max())
            if s > 1e-9:
                axis2, _ = project(np.array([c - s * g_hat, c + s * g_hat]), cam)
                if float(np.hypot(*(axis2[1] - axis2[0]))) > 1e-6:
                    lo = max(t - grad_amp, 0.0)
                    hi = min(t + grad_amp, 1.0)
                    grad = Gradient.from_ramp(ramp, tuple(axis2[0]),
                                              tuple(axis2[1]),
                                              t_range=(lo, hi), n=stops)
    return (float(depth.mean()), FilledCurve(
        pts2, role=Role.CONTENT, opacity=1.0, outline=False,
        color=ramp(t), gradient=grad, grain=grain,
        edge_color=edge, edge_width=edge_width if edge else None))


def extrude_items(poly2: np.ndarray, z0: float, z1: float, cam: Camera,
                  ramp: Ramp, *, side_grad_amp: float = 0.12,
                  cap_grad_amp: float = 0.12, bands: int | None = None,
                  grain: float = 0.0, edge: str | None = None,
                  edge_width: float = 0.35) -> list[tuple[float, FilledCurve]]:
    """A CCW polygon (viewed from +z) extruded from z0 to z1.

    bands quantizes the SIDE ramp only (the posterized-cylinder look);
    caps keep the smooth ramp. side_grad_amp=0 gives flat facets.
    """
    poly2 = np.asarray(poly2, dtype=float)
    side_ramp = quantize(ramp, bands) if bands else ramp
    kw = dict(grain=grain, edge=edge, edge_width=edge_width)
    items: list[tuple[float, FilledCurve]] = []
    m = len(poly2)
    for i in range(m):
        a, b = poly2[i], poly2[(i + 1) % m]
        quad3 = np.array([[a[0], a[1], z0], [b[0], b[1], z0],
                          [b[0], b[1], z1], [a[0], a[1], z1]])
        it = face_item(quad3, cam, side_ramp, grad_amp=side_grad_amp, **kw)
        if it is not None:
            items.append(it)
    top = np.column_stack([poly2, np.full(m, z1)])
    bot = np.column_stack([poly2, np.full(m, z0)])[::-1]
    for cap in (top, bot):
        it = face_item(cap, cam, ramp, grad_amp=cap_grad_amp, **kw)
        if it is not None:
            items.append(it)
    return items


def box_items(center, size, cam: Camera, ramp: Ramp,
              **kw) -> list[tuple[float, FilledCurve]]:
    """An axis-aligned box: center (x, y, z), size (sx, sy, sz)."""
    (cx, cy, cz), (sx, sy, sz) = center, size
    rect = np.array([[cx - sx / 2, cy - sy / 2], [cx + sx / 2, cy - sy / 2],
                     [cx + sx / 2, cy + sy / 2], [cx - sx / 2, cy + sy / 2]])
    return extrude_items(rect, cz - sz / 2, cz + sz / 2, cam, ramp, **kw)


def cylinder_items(center, radius: float, height: float, cam: Camera,
                   ramp: Ramp, *, facets: int = 48,
                   **kw) -> list[tuple[float, FilledCurve]]:
    """A vertical cylinder approximated by `facets` flat side quads.

    Sides are flat-shaded per facet (side_grad_amp=0): smoothness comes
    from facet count, the posterized look from bands=n.
    """
    cx, cy, cz = center
    th = np.linspace(0.0, 2.0 * np.pi, facets, endpoint=False)
    poly = np.column_stack([cx + radius * np.cos(th),
                            cy + radius * np.sin(th)])
    return extrude_items(poly, cz - height / 2, cz + height / 2, cam, ramp,
                         side_grad_amp=0.0, **kw)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_solids.py -v`
Expected: 6 PASS. If `test_box_visible_faces` reports 2 or 4 faces, the culling threshold or winding is wrong — fix the code, not the test (a generic camera with elev=32, azim=-35 sees exactly 3 faces of an axis-aligned box).

- [ ] **Step 5: Commit**

```bash
git add src/figlib/solids.py tests/test_solids.py
git commit -m "solids: extruded prisms/boxes/cylinders with light-aligned face gradients"
```

---

### Task 7: Demo figure + regression baseline

**Files:**
- Create: `figures/demo_solids_gradient.py`
- Modify: `figures/out/` (generated svg+png via --update)

**Interfaces:**
- Consumes: everything above; `surface3d.compose/as_floor/drop_shadow`; figure-program conventions (CLAIM/THEME/FORMAT/PARAMS/compute/build/assertions — mirror `figures/vca_fig14_volcanoes.py` and `figures/demo_glyphs_annulus.py`).
- Produces: a gate-passing, regression-tracked demo.

- [ ] **Step 1: Write the figure program**

Mirror the structure of `figures/vca_fig14_volcanoes.py` (Scene construction, ground plane via `as_floor`, camera from PARAMS). Content:

```python
"""Three solids under one light: the chromatic shading capability demo.

A flat face under a directional light has constant Lambert shade; the
within-face drift here is deliberate style — each face's gradient runs
along the light direction projected into the face plane, drifting a
fixed amplitude around the face's true Lambert tone. Shadow ends rotate
cool, lit ends rotate warm (OKLCh), the reference-image mechanism.
"""

import numpy as np

from figlib.format import COLUMN
from figlib.scene import Scene
from figlib.shading import chroma_ramp
from figlib.solids import box_items, cylinder_items
from figlib.surface3d import Camera, as_floor, compose, drop_shadow, surface_items
from figlib.theme import RISO

THEME = RISO
FORMAT = COLUMN

CLAIM = (
    "Three solids under a single light: each visible face carries its "
    "Lambert tone, drifted along the projected light direction with cool "
    "shadows and warm lights (OKLCh); the banded cylinder posterizes the "
    "same ramp; contact shadows tie each solid to the floor."
)

PARAMS = {
    "azim": -35.0, "elev": 32.0,
    "box_lo": {"center": (0.0, 0.0, 0.55), "size": (1.5, 1.5, 1.1)},
    "box_hi": {"center": (0.15, -0.1, 2.35), "size": (0.9, 0.9, 0.9)},
    "cyl": {"center": (1.55, 1.15, 0.45), "radius": 0.55, "height": 0.9},
    "bands": 6,
    "grain": 0.35,
    "floor_half": 2.6,
}


def compute(p):
    return {"params": p}


def build(g):
    p = g["params"]
    cam = Camera(p["azim"], p["elev"])
    # ramps anchored on theme palette hues — adjust hue signs by eye
    ramp_warm = chroma_ramp("#c0504d", hue_cool=-50.0, hue_warm=45.0)
    ramp_cool = chroma_ramp("#3a6ea5", hue_cool=-35.0, hue_warm=55.0)
    ramp_gold = chroma_ramp("#b08b2e", hue_cool=-60.0, hue_warm=30.0)

    h = p["floor_half"]
    floor3 = np.array([[-h, -h], [h, -h], [h, h], [-h, h]], dtype=float)
    ground = as_floor(surface_items(
        floor3[:, 0].reshape(2, 2), floor3[:, 1].reshape(2, 2),
        np.zeros((2, 2)), cam, dark=THEME.paper[1], lite=THEME.paper[0],
        edge_width=0.0))

    def verts(center, size):
        (cx, cy, cz), (sx, sy, sz) = center, size
        return np.array([[cx + dx * sx / 2, cy + dy * sy / 2, cz + dz * sz / 2]
                         for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)])

    lo, hi, cyl = p["box_lo"], p["box_hi"], p["cyl"]
    items = compose(
        ground,
        drop_shadow(verts(**lo), cam),
        drop_shadow(verts(**hi), cam),
        drop_shadow(verts(cyl["center"], (2 * cyl["radius"],) * 2 + (cyl["height"],)), cam),
        box_items(**lo, cam=cam, ramp=ramp_warm, grain=p["grain"]),
        box_items(**hi, cam=cam, ramp=ramp_cool, grain=p["grain"]),
        cylinder_items(**cyl, cam=cam, ramp=ramp_gold,
                       bands=p["bands"], grain=p["grain"]),
    )
    s = Scene()
    s.items.extend(items)
    # set xlim/ylim from projected extents the way vca_fig14 does
    return s


def assertions(g):
    pass  # geometry is synthetic; the gates and the eye are the checks
```

The implementer MUST first read `figures/vca_fig14_volcanoes.py` end to end and match its actual Scene/xlim/ylim/build-return conventions (the skeleton above is the content; the program contract comes from that file — e.g. whether `build` returns a Scene or mutates one, whether `assertions` is required).

- [ ] **Step 2: Render and iterate**

Run: `python -m figlib.cli figures/demo_solids_gradient.py`
Expected: gates pass. Then LOOK at `figures/out/demo_solids_gradient.png` (Read the file). Iterate on PARAMS/ramp knobs until: three distinct tones per box, visible within-face drift, banded cylinder, grain visible inside fills, shadows grounding all three solids. If a gate fails (contrast/expressivity), adjust ramp `l_range` — not the gate.

- [ ] **Step 3: Full suite + regression baseline**

Run: `python -m pytest tests/ -q` — all pass.
Run: `python -m figlib.cli --update figures/demo_solids_gradient.py` if the regress flow requires explicit baselining (check how other demos are baselined via `--regress`/`--update` in `cli.py`); then `python -m figlib.cli --regress` and confirm no drift in OTHER figures (this work must not change any existing figure's output).

- [ ] **Step 4: Commit**

```bash
git add figures/demo_solids_gradient.py figures/out/demo_solids_gradient.svg figures/out/demo_solids_gradient.png
git commit -m "demo: gradient-shaded solids still life (chromatic ramps, banding, fill grain)"
```

---

## Self-review notes

- Spec coverage: Gradient primitive (T2), renderer (T3), chroma ramps + quantize (T1), solids + face-gradient mechanism (T6), fill grain (T4), demo (T7), expressivity channel (T5), monotone-lightness invariant (T1 tests). Radial gradients emitted and tested (T3) though solids only use linear — the 2D wash follow-up needs radial.
- Deviation from spec, intentional: solids in `solids.py`, not `surface3d.py` (file-responsibility split; surface3d stays projection + heightfields).
- Two test helpers (`_svg_of`, `_survey_of`) are marked "mirror the existing test file" — the render/survey entry points are internal APIs whose exact names the implementer must copy from `tests/test_ink_channels.py` / `tests/test_expressivity.py`; the assertions are fully specified.
