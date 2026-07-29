# typical_set — friction record

## What the claim needed
A single y-axis carrying three quantities spanning ~40+ decades (C(N,r)
up to ~1e17, p^r(1-p)^(N-r) down to ~1e-31) so that their PRODUCT visibly
concentrates near O(1) while its two factors diverge in opposite
directions — the additive structure in log space (log(total) =
log(C) + log(p_seq)) is the actual mechanism, and it only reads as a
mechanism if all three curves share one honestly-labeled log axis rather
than three separately-normalized panels.

## What figlib gave for free
`plots.log10` as a `Scale` that composes with `plots.series`/`axis`
exactly like `Linear` — feeding raw (unlogged) values through
`yscale=` and letting the scale do `log10` internally kept `compute()`
numerically honest (the drawn values ARE the probabilities/counts, not
a pre-logged surrogate) while `tick_honesty` still checked the real
axis for clipping and padding. `Brace` for the measured typical-set
width, `Callout` for the "most probable sequence" provocation (leader +
box, no hand-rolled geometry), and `theme.categorical` for the two-factor
hue split — all reused without modification.

## What I hand-rolled
- `_sparse_log_ticks` (~10 lines) — `Log10.ticks()` places one label per
  decade by design (`step=` is explicitly rejected: "log10 ticks sit at
  decades"), which is correct for a 3-4 decade axis but unreadable at
  50+ decades. Building a custom `Ticks` at every k-th decade was the
  only way to get a legible axis at this dynamic range.
- Two rounds of deliberately re-deriving a scale's own layout: the
  x-axis needed asymmetric left/right padding (`r_pad_left` / `r_pad_right`)
  because a callout box needs real canvas room that a single symmetric
  `r_pad` couldn't provide without either clipping the box or padding
  the right side uselessly. Not a primitive gap — Scene xlim is exactly
  the knob — but it took a failed symmetric attempt first to see it.

## Gate diagnostics that did NOT contain the fix
The very first callout placement (anchored far right of its target to
dodge a label collision) passed every gate cleanly but produced a long
diagonal leader line that visually reads as a fourth data curve — no
gate flags this because a `Callout` leader is legitimate ink at any
angle; only inspecting the PNG caught it. The fix was architectural
(anchor the callout NEAR its target, pad the frame instead of relocating
across it), not a diagnostic the mechanical gate could have offered.

## Renders to first green: 12
## Proposed primitive
`plots.sparse_log_ticks(scale, step)` — every log-axis figure spanning
more than ~1 decade per label-width will hit the same
one-tick-per-decade default; this figure's version is generic (it only
reads `scale.range`), so it should move into `plots.py` rather than stay
duplicated per figure.
