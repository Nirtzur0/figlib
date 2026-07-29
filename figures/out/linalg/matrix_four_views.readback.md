# Readback record

**Verdict:** pass

## Intended claim

A matrix has four readings — one whole, mn entries, n columns, m rows — and choosing the column-and-row reading is what turns the product AB into a sum of rank-1 matrices, one per column of A paired with the matching row of B.

## Cold readback

Run against the pre-fix render (rank-1 marks unannotated, no dimensions).

**GLANCE** — "A 'four ways to read a matrix' figure — the same 3x2 array
shown as one block, as six scalars, as two columns, as three rows — with a
second row asserting that a matrix product decomposes into a sum of
rank-one column x row outer products."

**STUDY / claim** — "One matrix admits four equivalent readings, and
choosing the column/row reading makes matrix multiplication a sum of outer
products, AB = sum_k a_k b_k^T."

**Confusions raised**

- "The dimensions of A and B are never stated" — the bottom-left grey
  square is square while the top-left is 3x2 portrait, so the shapes are
  not the same matrix and conformability cannot be checked.
- "I had to guess which stripe is the column a_k and which is the row
  b_k^T"; the summand panels being all-blue "undercuts a rank-one product
  needing both."
- "In the first summand the leftmost bar is a lighter blue than the rest;
  I could not tell whether that lightness is meaningful (a weight, a
  highlight) or incidental."

**Verifiable vs. trusted** — verifiable: the counts (six dots = 3x2, two
vertical bands, three horizontal bands, shared aspect ratio and outline,
and exactly two summands matching the "2 columns" panel above). Trusted:
that the four top panels are the same matrix; that each summand is rank
one; that the two summands sum to AB; and dimensional compatibility.


## Notes

GLANCE recovered the claim, so macro-structure passes. Three of the four
confusions were real defects and were fixed; one is accepted.

**Fixed — the opacity encoding was undecodable.** The cold reader saw the
pale band and could not tell whether it meant anything. It did: `rank1`
sets each column's opacity to |b_k[j]|. The fix draws b_k^T itself as a
1 x n strip directly above each summand under the SAME opacity law, so
pale entry sits over pale column, blank over blank, solid over solid. The
encoding now documents itself on the page rather than in a docstring.

**Fixed — the row factor was missing.** "All-blue undercuts a rank-one
product needing both" was correct: only the column factor was drawn. The
b_k^T strip supplies the other one, in the row hue, restoring the two-hue
reading the top row sets up.

**Fixed — dimensions were never drawn.** The figure *gates* conformability
(`check_conformable`) but never showed it, so the reader could not perform
the check the program performs. m and n are now drawn on the edges of the
leading block of each row.

**Accepted — AB carries no structure to compare against.** The reader
cannot verify by eye that the two summands sum to AB; the grey wash is
deliberately the "unstructured whole" reading, and the arithmetic is
certified by `check_expr` (max|LHS-RHS| against A@B on the same arrays
that got drawn) rather than by inspection. Encoding AB's values as a
heatmap would mix a value encoding into a structure figure, which
grammar.md warns against, so this stays on trust by design.

**Accepted — "the four top panels are the same matrix" is on trust.** True
and intended: the top row's claim is about readings of one object, and
showing entry values would make the four panels differ in ink where they
must not. The shared outline, aspect ratio and counts carry it.

An earlier iteration of `rank1` painted one column and one row of the
RESULT block. That was a lie — the summand is the whole rectangle, not
that cross — and it was rewritten to the current "every column is a_k
scaled by b_k[j]" mark before this readback ran.

