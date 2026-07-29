# Readback record

**Verdict:** pass

## Intended claim

A smooth Gaussian kernel matrix is nearly rank 4: drawn exactly as K = sum_{k<=4} sigma_k u_k v_k^T + R, its four rank-1 modes are signed and oscillate at rising frequency like a Fourier basis, and the residual R is visibly flat at the same color scale as K — 5.6% relative Frobenius error, 99.7% of the energy kept.

## Cold readback

Run against the pre-fix render, whose expression row was `K_4 = sum_k
sigma_k u_k v_k^T` — K itself was not drawn.

**GLANCE** — "An 'equals plus plus plus' decomposition of a matrix into
four rank-one outer-product terms — asserting that a smooth kernel matrix
is (nearly) the sum of a handful of rank-one pieces."

**STUDY / claim** — "A smooth kernel matrix K admits an accurate low-rank
approximation — its truncated rank-4 SVD reproduces K to 5.6% relative
Frobenius error (99.7% of the squared energy)."

**Confusions raised**

- "The leftmost panel is labeled K_4, i.e. the approximation, not K
  itself — yet the error statement compares K to K_4, and K is never
  shown. So I cannot see the thing being approximated; the '=' is exact
  by definition, and the interesting residual is invisible."
- The Hinton strip "reads as one anonymous 4-row grid with no per-row
  labels, no axis, and no visible link back to the four panels."
- "'the bar scales every block above' is ambiguous: does the colorbar's
  range apply to all four term panels jointly ... or is each panel
  separately normalized?"

**Verifiable vs. trusted** — verifiable: the additive structure and term
count; the smooth dark diagonal band; that terms 2, 3, 4 carry
progressively finer structure ("roughly 2, 3, 4 sign lobes across the
panel - the standard oscillation-count signature of higher singular
vectors"); that colors are signed. Trusted: "the numbers 0.056 and 99.7%
(K is not drawn, no residual panel, no singular-value spectrum)"; that
the panels are outer products rather than arbitrary fields; that the four
panels sum to the left panel; and the decay sigma_1 >> sigma_4.


## Notes

GLANCE recovered the claim, so macro-structure passes. The first
confusion was a genuine design error and drove a restructure.

**Fixed — the thing being approximated was invisible.** Drawing
`K_4 = sum_k` is exact by definition, so the honesty move (draw a true
equation, admit the error as a number) had the side effect of hiding the
figure's actual subject. The fix makes the identity *more* exact, not
less: the residual is drawn as the last term, so the row is now
`K = sum_{k<=4} sigma_k u_k v_k^T + R`. K is on the page with its
diagonal band; R is rendered at the SAME symmetric scale as everything
else, and its near-flatness is now the claim — readable off the figure
instead of taken on trust. Two assertions were added with it: that R
really is K - K_4, and that max|R| < 0.25 max|K|, which is the figure's
central *visual* claim stated numerically so it cannot silently rot.

**Fixed — colorbar ambiguity.** "Does the bar apply jointly or per
panel?" It applies jointly, which is the whole point (per-block
normalization would rescale each summand to itself and make the sum look
wrong). The legend now says so explicitly, and with R in the row the
shared scale is self-evidencing: a per-panel normalization could not make
R look flat.

**Accepted — rank-1 structure is not verifiable by eye.** Correct, and
inherent to a color field: no row/column profile is shown. The rank-1
property is certified by an assertion on each drawn term instead. Adding
row and column profile strips per panel would quadruple the ink for a
property the reader already accepts from the sigma_k u_k v_k^T label.

**Accepted — no singular-value spectrum.** The reader is right that the
decay is "the actual content of low rank works here and is nowhere
plotted." It is now partly visible (R flat at the shared scale is exactly
the statement that what is left over is small), and a spectrum plot would
be a second figure with a different claim. `check_expr` at rtol 1e-12 and
the energy assertion carry the rest.

**Accepted — Hinton strip rows are unlabelled.** The strip is a secondary
inset; per-row labels at 9 pt would push the annotation census past the
load gate. The legend names it v_1..v_4 and the row order is the obvious
one.

