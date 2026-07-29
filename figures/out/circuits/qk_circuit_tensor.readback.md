# Readback record

**Verdict:** pass

## Intended claim

Attention's query and key projections are not two mechanisms: they meet along the head axis, so X W_Q W_K^T X^T and X W_QK X^T are the same array for W_QK = W_Q W_K^T — a single bilinear form of rank 3, the head width, inside an 8-dimensional model.

## Cold readback

(Third cold read, on the shipped render. Two earlier reads drove the fixes
below; their defects are listed in Notes.)

**GLANCE.** "A tensor-network / Penrose-diagram identity for attention:
two chained weight tensors between query and key inputs collapse into a
single low-rank QK matrix — the top and bottom diagrams are asserted
equal."

**CLAIM READ.** "Contracting X-W_Q-W_K-Y along the head dimension h is
exactly the same tensor as contracting X-W_QK-Y with W_QK = W_Q W_K^T, so
the attention logit map is a single rank-h bilinear form on
residual-stream space, with h = 3 < 8 the actual rank (verified
numerically to 1e-15)."

**CONFUSIONS.**
- "The transpose is invisible by design, which is the point being made —
  but I had to trust that the h-leg between W_Q and W_K is the shared
  inner index. Nothing in the picture forbids reading the chain as
  W_Q^T W_K."
- "The einsum strings carry information the geometry does not (that W_K
  is indexed eh, hence the transpose). A reader who skips them mis-reads
  the picture."
- "s = 5, t = 4 are declared but nothing in the figure varies with them;
  the legs are unadorned lines, so no dimension is visually encoded."

**VERIFIABLE BY INSPECTION.** "Both diagrams have exactly two dangling
legs (s, t), so both are s x t objects; the h-leg is internal in the top
and absent in the bottom, so it has been summed away; the index strings
are consistent with each other and with the stated shapes; that W_QK is
highlighted as the derived object."

**ON TRUST.** "That W_Q W_K^T is what the top contraction actually
evaluates to (the leg-joining convention is asserted, not demonstrated);
that rank W_QK = 3 rather than merely <= 3; the 1e-15 residual, which is
a reported numerical result with no visible evidence; that the s, t legs
are pre-softmax logits."


## Notes

**Verdict: pass.** The glance read now recovers the whole claim including
the low-rank fact, which is the bar for macro-structure.

**Fixed, from readbacks 1 and 2.**
- *W_QK was never defined on the page.* Reader 1 had to infer the
  transpose from the einsum strings. The bottom line now states
  `W_QK = W_Q W_K^T` explicitly.
- *The rank fact — the actual point — was never printed.* Reader 1: "that
  low-rank fact is arguably the whole point, but the figure never states
  it; I had to guess whether the numbers were illustrative or
  load-bearing." Now printed, and printed as a MEASUREMENT
  (`matrix_rank(W_QK)`), because reader 2 correctly objected that a
  printed `rank = h` is a claim while a computed one is evidence.
- *Scope was unstated.* Neither reader could tell whether the softmax was
  missing or elsewhere. A header line now names X and Y and says the
  dangling legs are the logits, before softmax.
- *The `=` read as a margin annotation, not a relational operator* — said
  by both readers. It was in the left margin, vertically between the rows
  but horizontally beside neither. It now sits on the spine, and the page
  reads top to bottom as one equation: diagram / its einsum / = /
  diagram / its einsum.
- *The einsum strings floated between the rows* with "no explicit tie to
  which diagram each belongs to" (reader 2). `spec_drop` halved so each
  string hugs its own row.
- *The residual floated unattached* and, separately, "Delta between what
  and what" was never said. It now rides on the identity line it
  certifies and names its operands: `max|row 1 - row 2|`.
- *`1e-15` typeset as math read as a subtraction.* Set as `10^{-15}`.
- *Index names were upright in the header and italic on the legs* — two
  alphabets for one correspondence. Unified.
- *The free-leg labels were struck through by their own stubs.* Fixed in
  `tensor.py`: an index label now sits PAST the tip with ha/va pointing
  away, the `schematic.Junction` arg-label idiom. Test:
  `test_a_free_legs_label_sits_past_its_tip_not_on_the_line`.

**Accepted, with reasons.**
- *"Nothing in the picture forbids reading the chain as W_Q^T W_K."*
  True, and it is the notation's known cost rather than this figure's
  bug: in Penrose notation a leg carries no orientation mark, which is
  exactly the fact the figure states on the page ("a transpose IS which
  leg joins"). The remedy in the literature is to name axes instead of
  ordering them (Named Tensor Notation, arXiv 2102.13196), which is what
  `Leg.index` does — the wire is labelled `h`, and `dims()` raises if a
  tensor's leg count disagrees with its array rank. **The honest limit:
  swapping two axes of EQUAL length would pass every gate.** Here d = e =
  8, so a d/e swap on W_QK is not caught by shape; it IS caught by
  `check_einsum`, which evaluates against `scores`.
- *"The 1e-15 residual has no visible evidence."* Correct, and
  unfixable by drawing: the demonstration is `check_einsum` plus
  `np.allclose` on the two drawn networks, which is stronger than a
  worked example the reader would have to trust anyway. The number on the
  page is the gate's number.
- *"No dimension is visually encoded on the legs."* Deliberate. Encoding
  a dimension as leg thickness or count would make the picture
  quantitative in a notation where nothing else is drawn to scale, and
  `matrix.py` already owns shape-as-geometry. The dims are gated
  (`check_index_dims`, `check_output`) and printed once.

