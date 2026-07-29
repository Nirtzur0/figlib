# Comparative judge record — demo_sphere_stereographic

**Verdict (final): RECREATION BETTER** (vs VCA Fig [19], p.140,
refs/vca_fig19_stereographic-161.png)

Round 1 (secant redesign + final fix round, judged from both PNGs):
RECREATION BETTER — keeps every device of the book figure (opaque sheet
through the equator with the southern cap poking below its near edge,
UNIT CIRCLE set in perspective on the sheet, two sample rods from N with
chords dashed through the interior, hollow arrows chasing p to infinity,
axis dots re-emerging near S) and adds the fixed-set argument on-figure:
L is drawn as a secant, its two unit-circle crossings are marked, and the
image circle passes through the same two marked points via exact
parameter samples (asserted to 1e-9, not eyeballed). The book leaves
"equator points are fixed" entirely to the text. Occlusion is honest
throughout: the image circle's southern dip is dashed by exact sight-line
tests against the near sheet, verified in assertions().

Caveats (book better at): the engraved stipple shading gives the book's
Sigma more tactile volume than the flat printed-tan body; the book's
rods read as physical wires (double-stroked tubes) where ours are
colored strokes.

Fix-loop dispositions this round:
- CIRCLE label on the sphere limb -> arc-word radius 1.40 -> 1.52
  (halo and off-arc relocation rejected; device preserved). FIXED.
- t=0.6 line-stub arrowhead on the CIRCLE label -> stub dropped; one
  hollow arrow per solid run of L remains. FIXED.
- figcheck: PASS. Golden regression: MATCH.
