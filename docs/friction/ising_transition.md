# ising_transition — friction record

## What the claim needed

An order-parameter-vs-control-parameter curve (the Onsager |m(T)|,
exact, zero above T_c) plus three bound instances — sampled 64x64 spin
configurations at deep-ordered, critical, and deep-disordered T — where
spin is a two-level CATEGORICAL field (never an ordered ramp), the three
snapshots are tied to their T on the curve by rails + tags (the
saddle-node exemplar's grammar, read top-to-bottom instead of
bottom-to-top), and the one thing that could silently be wrong — an
under-equilibrated Metropolis sampler producing noise at every T instead
of a real ordered/critical/disordered hierarchy — is gated numerically.

## What figlib gave for free

- `RasterField` with `(H, W, 3)` sRGB input was exactly the primitive for
  a 64x64 spin lattice: no per-spin scene items, one image per panel.
  `theme.categorical(0)`/`categorical(1)` → `color.to_rgb()` → broadcast
  over the boolean spin mask gave the two-level hue with zero new
  machinery.
- `Figure(grid=(2, 3))` with 3 panels in row 0 and 1 panel (alone in its
  row, so it takes the full row width) in row 1 reproduced the
  saddle-node exemplar's small-multiples-then-family layout exactly —
  no new layout code, just panel order.
- `plots.axis`, `plots.series`, `plots.markers`, `plots.tick_honesty`
  covered the whole order-parameter panel; `gates.Checks` gave one gate
  run reporting every numerical defect instead of stopping at the first.

## What I hand-rolled

- The vectorized checkerboard-parity Metropolis sweep (`_ising_sweep`,
  ~15 lines) and the burn-in/sample loop (`_sample_configuration`,
  ~12 lines). Not a figlib gap — this is genuinely figure-specific
  numerics (compute(), not drawing) — but it's the kind of "sample a 2D
  Ising lattice" device that will recur if this corpus ever needs
  another statistical-mechanics figure (Potts, XY, percolation).

## Gate diagnostics that did NOT contain the fix

- First render came back with the ENTIRE canvas painted solid brick-red
  (`Role.ACCENT2`), including the panel that should have held the
  order-parameter curve. The `expressivity` note ("heaviest: FilledCurve
  ACCENT2 100%") pointed at the right object but not the cause. The
  actual bug: `plots.markers(..., size=(6.0, 6.0))` — I'd passed a
  canvas-px constant directly as the SCENE-UNIT marker size, on an axis
  whose T-span is only ~3.2 units, so each marker's circumradius was
  ~2x the entire plot. No gate flags "marker radius exceeds axis span";
  it just renders a giant disc and the mechanical gates (which check
  label/collision boxes, not fill area) had nothing to say. Fix required
  recognizing the saddle-node exemplar's `DOT_PX * upx` pattern (px size
  converted to math units via the panel's own `Transform.scale_x`) and
  restructuring `build()` to populate scenes AFTER `layout_figure` has
  fixed their transforms, exactly as the exemplar does — this is
  documented in architecture.md but easy to skip when a figure "looks
  done" after the numerical gate alone passes on the first attempt (the
  numerical assertions never touch drawn marker *size*, only position).
- A `stale-change` residual on first use of `CORRESPONDENCE`: I gave the
  three lattice `RasterField`s a shared key and declared it in
  `changes=`, expecting the gate to diff the actual pixel content. It
  doesn't — `correspond.py`'s residual check compares `(type, role,
  color, dash)` metadata, not `values`, so three genuinely different
  64x64 arrays under one key read as "identical, so the declared change
  never happened." The diagnostic's fix (`add to changes=`) was already
  what I'd done, so it was actively misleading here; the real fix was
  recognizing these three panels are independent draws with no shared
  object at all, and dropping `CORRESPONDENCE` entirely (skill.md is
  explicit that this is a valid — and silent — outcome, but the gate
  message reads as if some declaration would fix it).

## Renders to first green: 3

(1st: `label-on-ink` on the T_c label + a `stale-change` residual on the
premature `CORRESPONDENCE`, plus a bad "near T_c" continuity assertion
that failed on real physics — critical exponent beta=1/8 makes the
approach to 0 genuinely slow, not a bug. Fixed the label position,
dropped `CORRESPONDENCE`, replaced the assertion with an epsilon-limit
check. 2nd: passed the gate, but the render was the giant-marker
solid-red canvas above — gates don't check drawn area, only the PNG
caught it. 3rd, after the `DOT_PX * upx` fix and the
populate-after-layout restructure: clean, `[PASS]`, matches intent.)

## Proposed primitive

None for the marker-size bug specifically — the exemplar pattern
(`layout_figure` before populating, `DOT_PX * upx`) already exists and
is documented; the gap is discoverability, not missing code. If a
primitive is worth adding: `gates.py` could sanity-check that a Point/
`FilledCurve` marker's rendered bbox is a small fraction (say < 5%) of
its panel's canvas area — the kind of gross unit-confusion (px passed
where scene-units were expected, or vice versa) that produced 100%-ink
canvases here and is otherwise invisible to every existing mechanical
check.
