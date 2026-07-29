# Chromatic gradient shading + 3D solids

Date: 2026-07-29
Status: approved design, pre-implementation

## Goal

Extend the shading system toward the reference look (grainy multi-hue
gradients on floating boxes/cylinders): shading that travels through hue,
genuine per-face gradients, grain scoped inside fills — as a *general*
capability (2D fills included), not a 3D-only effect. Plus first-class 3D
solid primitives so the look has plot types to live in.

## Non-goals

- Raster/per-pixel shading (RasterField route): abandons SVG-native
  identity, breaks ink/expressivity accounting. Rejected.
- Dense-faceting fake gradients: file bloat, seams, no reusable channel.
  Rejected.
- feTurbulence grain: cairosvg filter support unreliable. Grain reuses the
  proven tile-pattern route.

## Components

### 1. `Gradient` primitive — scene.py

```python
@dataclass(frozen=True)
class Gradient:
    stops: tuple[tuple[float, str], ...]  # (offset in [0,1], "#rrggbb")
    kind: str = "linear"                  # "linear" | "radial"
    p0: XY = (0.0, 0.0)                   # math coords: linear axis start / radial center
    p1: XY = (1.0, 0.0)                   # linear axis end / radial edge point
```

`FilledCurve.gradient: Gradient | None = None`. When set, it is the fill
paint (overrides `color`). `gradient` + `pattern` together is a
`ValueError` — competing paints.

### 2. Renderer — render.py

- `<linearGradient>` / `<radialGradient>` defs with
  `gradientUnits="userSpaceOnUse"`; `p0/p1` pass through the layout
  `Transform` like all geometry. Deduped by content hash (same idiom as
  `_ensure_pattern`).
- cairosvg renders both gradient kinds natively; PNG parity holds.

### 3. Chromatic ramps — shading.py (new module)

OKLCh on top of color.py's OKLab.

- `chroma_ramp(base, *, l_range=(l_dark, l_light), hue_cool, hue_warm,
  c_boost=0.0) -> Callable[[float], str]` — t=0 shadow: hue rotated
  toward cool, lightness at l_dark; t=1 lit: toward warm, l_light.
  Gamut mapping by chroma reduction at fixed (L, h) — never RGB clipping,
  which shifts hue.
- `quantize(ramp, bands) -> ramp` — posterized steps (reference cylinder).
- Ramps are drop-ins for the existing `shade=` hook of `surface_items`,
  so heightfields/sphere inherit the chromatic model unchanged.

Invariant (tested): ramp lightness is monotone in t — "lightness is
order" applies to shading ramps too.

### 4. Solids — surface3d.py

- `box_items(center, size, cam, ramp, ...)`,
  `cylinder_items(center, radius, height, cam, ramp, facets=48,
  bands=None, ...)`, `prism_items(polygon, height, cam, ramp, ...)`.
- Faces backface-culled by normal · toward; emitted as depth-tagged
  FilledCurves; existing `compose` / `drop_shadow` / `as_floor` apply.
- Within-face gradient mechanism: a flat face under directional light has
  constant Lambert t, so the reference's per-face ramps are stylistic.
  Each face gets a linear gradient whose axis is LIGHT_DIR projected into
  the face plane then to screen, ramping ramp(t−δ) → ramp(t+δ) around
  the face's Lambert value. δ = style knob (`grad_amp`), default modest.
  Cylinder side facets stay flat-shaded; `bands=n` quantizes them.

### 5. Fill-scoped grain — scene.py + render.py

`FilledCurve.grain: float = 0.0`. When > 0 the renderer emits a second
copy of the path filled with the existing grain tile pattern at that
opacity, clipped to the shape. Skipped on transparent themes.

### 6. Demo — figures/demo_solids_gradient.py

Reference-style still life: two floating boxes + banded cylinder,
chromatic ramps from theme palette, drop shadows, fill grain. Full
gates + golden regression like every figure.

## Testing

- chroma_ramp: monotone OKLab lightness in t; endpoints in gamut;
  quantize produces exactly n distinct colors.
- Gradient defs: dedup (two identical gradients → one def); coords are
  canvas-transformed.
- Solids: backface culling (hidden faces absent), depth order sane for a
  reference camera.
- FilledCurve: gradient+pattern raises.
- Expressivity: gradient registers as a channel.

## Open follow-ups (out of scope)

- 3D prism chart (bar field) as a real data plot using solids.
- Gradient washes retrofitted into 2D figures (basin wash, ODE/SDE).
- Reshading vca_fig14 / sphere with chroma_ramp.
