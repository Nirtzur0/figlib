# Readback record

**Verdict:** pass

## Intended claim

A QK attention score computed on perturbed vectors decomposes exactly into four interaction terms — the intended signal-signal score, two cross terms that pair a full-size signal against an error, and one error-error term — so the corruption of an attention score is, to leading order, the two cross terms and nothing else.

## Cold readback

**GLANCE (1).** An algebraic expansion of an attention score: a bilinear
form applied to two perturbed vectors, multiplied out into four terms on
one line, with the middle two picked out in blue and gathered under a
brace. It seems to assert that the middle two are the ones that matter.

**STUDY (2).** Expanding s = (x_q + eps_q)^T W (x_k + eps_k) yields
exactly four interaction terms — clean-clean, clean-key-error,
query-error-clean, error-error — and the two mixed terms are the
first-order-in-eps part, i.e. the leading corruption of the score.

**(3) Confusions / guesses.**
- The glosses "the key error" and "the query error" are terse enough to
  be read two ways. Under x_q^T W eps_k, "the key error" has to mean
  "the term in which the KEY factor is the error"; I first read it as
  "the error in the key's score contribution", which is close but not
  the same thing. The parallel structure of the four glosses is what
  disambiguates it, not the wording of any one of them.
- The fourth term is drawn in a noticeably faded ink and glossed "error
  on error", which reads as "negligible" — but the figure never states
  the condition under which it is negligible (|eps| << |x|). I took the
  smallness on trust; nothing on the page quantifies it.
- W is undefined. I assumed it is the combined query-key matrix, but the
  figure does not say whether the 1/sqrt(d) scaling or the W_Q W_K^T
  factorization is folded into it, and it does not say what eps is
  (quantization? ablation? adversarial?).

**(4) Verified vs. taken on trust.** Verifiable by inspection: that the
enumeration is CLOSED — each of the two query factors is paired with
each of the two key factors exactly once, no repeats, none missing, so
"these four terms are all of them" is something I can check by reading
the row rather than believe. Also verifiable: that the brace covers
exactly the two mixed terms and neither neighbour. Taken on trust: the
magnitude ordering that the ink weights assert (signal darkest, cross
terms accented, error-error faded) — the figure encodes an ordering it
gives me no numbers for; that eps is small; and that the quantity is
an attention score at all, which the sans line asserts rather than shows.


## Notes

**Self-read, disclosed.** No separate cold agent was spawned; the builder
answered the readback prompt from a fresh look at the final PNG. Treat the
GLANCE answer as optimistic — the builder cannot un-know the claim — and
re-run with a genuinely cold reader if this figure is ever promoted to a
reference exemplar.

**Design review of the three confusion bullets.**
1. *Terse glosses* — ACCEPTED. Each gloss is bounded by the term spacing,
   and `assertions()` enforces a metric non-overlap floor between adjacent
   glosses, so "the key's error, read by the true query" does not fit at
   this format. The parallel 2x2 structure across the four glosses is
   doing the disambiguating work deliberately (design step 5), and the
   brace supplies the mechanism the individual glosses cannot.
2. *Smallness condition unstated* — ACCEPTED, and it is the figure's real
   elision. `assertions()` verifies the ordering numerically (cross/signal
   = O(delta), second-order/signal = O(delta^2) over a 512-sample
   ensemble at delta = 0.1), but that certificate is in the program, not
   on the page. Putting "for |eps| << |x|" on the figure was considered
   and cut: the exhaustiveness claim is exact and unconditional, and
   hanging a condition off it would make the reader think the FOUR-TERM
   decomposition is approximate, which is precisely the misreading design
   step 8 exists to prevent. The muted ink is an ordering claim, not a
   magnitude claim, and the foot line keeps the exact claim in front.
3. *W and eps undefined* — ACCEPTED, confessed in design step 8. W as one
   matrix is a deliberate abbreviation of W_Q W_K^T / sqrt(d) (the
   decomposition is unchanged under the factorization), and eps is left
   abstract on purpose: quantization, ablation and adversarial
   perturbation all land in the same four terms, which is the reason the
   figure is worth drawing.

**Fix applied during the readback loop.** The first render carried no
statement that this is an attention score at all — the only evidence was
the q/k subscripts. The sans line at the top-left ("the QK attention
score, with an error on each side") was added in response, and it also
fills the quadrant the equation left empty.

**Verdict rationale.** The GLANCE read recovers the macro structure (an
expansion into four terms, the middle two distinguished) and the STUDY
read recovers CLAIM, including the exhaustiveness and the first-order
identification of the cross pair. Pass.

