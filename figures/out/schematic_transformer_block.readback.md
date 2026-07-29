# Readback record

**Verdict:** pass

## Intended claim

In a pre-norm transformer block the residual stream is the main path and nothing sits on it: attention and the MLP are branches that READ a normalized copy of the stream and ADD their output back, so the LayerNorms never touch what is carried forward and every write is the same d_model width as the stream it is added to.

## Cold readback

GLANCE: A vertical flow diagram of one transformer layer: one thick
arrow runs straight up the middle through two circled-+ nodes, with a
LayerNorm -> Multi-Head Attention loop hanging off the left and a
LayerNorm -> MLP loop off the right. The thick line looks like the point:
the main path is the stream itself, and the sublayers are side loops.

STUDY:
2. Claim: across a (pre-norm) transformer block the residual stream is
   carried forward unchanged except for two additive writes — attention
   and the MLP each read a LayerNormed copy of the stream and add their
   d_model-wide output back at a circled +, so no LayerNorm (indeed no
   sublayer at all) sits on the path that is carried forward, and the
   same structure continues upward into further layers.
3. Confusions / guesses:
   - The stacked cards behind Multi-Head Attention read as "several of
     these in parallel" — presumably the heads — but the count is
     unstated, and a cold reader could instead take it as "this box
     repeats across layers".
   - The hollow diamond under the top ellipsis is an unfamiliar glyph; the
     "continues beyond the drawing" reading comes almost entirely from the
     ellipsis, not the diamond.
   - Hollow vs filled arrowheads clearly encode something (transformed
     output vs the stream carrying itself?) but there is no legend; and
     d_model is written on the stream and on the MLP write, so the
     attention write being d_model too has to be inferred by symmetry.
4. Verifiable by inspection: each branch taps the stream at a dot BELOW
   the + where it writes back; both LayerNorms sit inside the dashed
   sublayer groups, off the central line; the thick line never enters any
   box; the two branches are on opposite sides. On trust: that circled +
   is vector addition, that the two writes really have the stream's
   width, and everything inside the boxes.

GLANCE GRADE: the glance read lands on the claim (thick central line =
the residual stream is the main path; sublayers are loops off it).

## Notes

Cold read performed by the task agent from a fresh look at the PNG
(no separate harness available); misses recorded, not smoothed.
Accepted ambiguities: the stack mark's head-count is inherently
unstated (that is what an abbreviation mark means); the truncation
diamond is carried by its ellipsis for a reader who has not seen the
mark before — acceptable, the ellipsis is the load-bearing glyph; the
hollow/filled head distinction remains legendless by house grammar
(kind vocabulary), unchanged from the previous record.
