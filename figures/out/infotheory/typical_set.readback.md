# Readback record

**Verdict:** pass

## Intended claim

For N i.i.d. Bernoulli(p) draws with p < 1/2, the all-zeros sequence (r = 0 ones) is individually the most probable single sequence, but it lies outside the typical set -- the band of width O(sqrt(N)) centered at r = pN that carries essentially all the probability mass -- because the multiplicity C(N,r), which rises toward r = N/2, and the per-sequence probability p^r(1-p)^(N-r), which falls monotonically, trade off there and nowhere else.

## Cold readback

GLANCE: This is about probability concentration in a binomial/Bernoulli-sequence
setting -- it seems to argue that the "typical" outcomes (a narrow band around the
mean) dominate the total probability mass even though no single typical sequence
is anywhere near the most probable one.

STUDY -- the claim: for i.i.d. Bernoulli(p) sequences of length N, the
exponentially rare per-sequence probability p^r(1-p)^(N-r) is overwhelmed by the
combinatorially huge multiplicity C(N,r), so their product (total probability
mass) peaks not at r=0 (the single most probable sequence) but at r = pN, in a
band of width O(sqrt(N)) -- i.e., "typical" is a statement about aggregate mass
concentrating away from the most probable individual outcome, not about any
single outcome being likely.

Confusions / ambiguities:
- The red "most probable single sequence" callout sits at r=0 with a value near
  10^-10, but it's not obvious why that particular value follows from N=60,
  p=0.30 without doing the arithmetic -- the figure asserts it rather than
  showing the computation.
- The gray shaded band is labeled "typical set" with bounds r in [7,29] and
  width 2x3sigma, but the boundary lines don't crisply align with where the blue
  "total mass" curve visually starts/stops looking flat -- had to trust the
  stated numbers over the visual cue.
- It's a bit ambiguous whether the dashed vertical line at r=pN=18 is meant to
  mark the peak of the blue curve or just the mean -- visually the blue curve's
  apparent peak looks slightly left of that line.

Verifiable by inspection: the three curves' overall shapes and relative
ordering (multiplicity rising to a huge peak, per-sequence probability falling
monotonically and steeply, total mass as a product peaking in between); the
shaded band's stated endpoints roughly bracketing the region where the blue
curve is near its max; the axis being log-scale and spanning ~40 orders of
magnitude. Taken on trust: the exact numeric values (sigma=3.55, the r in
[7,29] endpoints, the red point's value ~1e-10), and the claim that this is a
general asymptotic phenomenon (O(sqrt(N)) width) rather than specific to
N=60, p=0.3.

## Notes

GLANCE and STUDY both land exactly on the intended claim, unprompted --
including the precise mechanism (multiplicity vs. per-sequence probability
trading off) and the O(sqrt(N))-band-not-the-mode punchline.

Two of the three confusions are the honest cost of a quantitative claim on a
40+ decade log axis: the reader cannot re-derive C(N,r) p^r (1-p)^(N-r) from a
picture, and is not meant to -- that arithmetic is exactly what assertions()
gates instead of asking the reader to trust it by eye. Accepted as-is.

The third point (dashed r=pN line looking slightly left of the blue curve's
apparent peak) was checked against the data: argmax_r of the drawn total-mass
array is exactly r=18=pN for N=60, p=0.30, so the two coincide exactly and the
mismatch is a rendering/perception artifact (curve curvature near a broad,
flat-topped peak reads as ambiguous by eye), not a drawn error. Accepted --
the guide line is correct; nothing to fix.
