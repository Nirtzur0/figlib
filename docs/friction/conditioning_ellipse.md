# conditioning_ellipse — friction record

## What the claim needed
A domain circle and codomain ellipse, side by side, with three matched
directions (v1/v2/generic on the input, their images on the output) and a
numeric κ = σ1/σ2 annotation, using Figure/Panel/Connector for the map
arrow between them.

## What figlib gave for free
Figure/Panel/Connector gave the two-panel-plus-map-arrow layout, tag
boxes, and independent per-panel Transforms (so a 1-unit circle and a
2.8-unit-radius ellipse each get their own correct scale) with zero
manual layout math. `autoplace` cleared every label collision inside each
panel automatically (7 nudges, all sane). The mechanical gate's
`label-on-ink` and `clipped` diagnostics pinpointed exactly which labels
needed shorter text or outward-biased placement.

## What I hand-rolled
- `_outward(tip, pad)` (~10 lines): ha/va/offset_px chosen from the sign
  of a vector's tip coordinates, so a label anchored at an arrowhead
  grows away from the origin instead of back over the shaft. Every figure
  with radiating vectors from a common origin needs this; it isn't
  vector-specific plumbing, it's the same "outward" rule fig09's spiral
  labels hand-tuned per-label instead of computing.

## Gate diagnostics that did NOT contain the fix
The composite (`CORRESPONDENCE`) machinery actively fought this figure
and I removed it rather than fight back further. The binding here is
"the same vector, before and after A" — but `correspond.py`'s fingerprint
deliberately excludes position/length (documented: "moving is usually
the claim"), so every keyed vector registered as facet-identical across
panels (same role, same color, same type) regardless of how much its
length changed. That makes any key I put in `changes=` fire
`stale-change` ("declared changing but identical"), while leaving it out
makes the gate treat it as a claimed *invariant* and fire
`fixed-set-rescaled` when the two panels are — correctly — drawn at
different page scales (domain radius 1, codomain semi-axis 2.8). There is
no `changes=`/`frame=` combination that satisfies both checks when the
thing that changes is a magnitude rather than a facet, because magnitude
isn't part of what `changes=` can even name. The diagnostic text pointed
at plausible-looking fixes (state a rescale in `frame=`) that turned out
to be dead ends for this shape of claim — I only found the real
resolution (don't key these items; the correspondence gate is scoped to
facet-changing bindings, not magnitude-changing ones) by reading
`correspond.py`'s source, not from the diagnostic.

## Renders to first green: 3
(fail 1: clipped/label-on-ink/stale-change/frame-drift; fail 2:
fixed-set-rescaled after fixing the first batch; pass 3 after dropping
the correspondence declaration.)

## Proposed primitive
`figlib.scene.label_outward(tip: XY, pad: float) -> dict[str, Any]` — the
ha/va/offset_px-from-quadrant helper, as a shared function instead of a
per-figure `_outward`. Signature only; "none" would also be defensible
since it's 10 lines, but it recurs (fig09_exp_series_spiral hand-picks
per-label offsets doing exactly this by eye) and a shared version would
have saved one of the two failed render cycles here.
