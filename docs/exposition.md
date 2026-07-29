# Exposition: the thinking before the figure

grammar.md says how to render a figure honestly. This document is
upstream of it: why a figure works at all, what deserves one, how to
choose the representation, and how a reader's minute of attention gets
spent. Sources: a close read of Needham's method (preface manifesto +
figure-by-figure study of Ch. 1, 4, 7), Larkin & Simon 1987, Thurston
"On Proof and Progress", Tufte, Mayer's measured effects, Nielsen,
Victor, Olah/Carter, Nelsen, Doumont. Everything below is a decision
rule, not taste.

## Why a figure works (the mechanism)

Two representations can carry identical information and still differ
completely in what inferences are cheap — Larkin & Simon call this
**informationally equivalent but computationally inequivalent**, and it
is the whole theory. Prose is indexed by sequence: the premises of an
inference are scattered, and the reader pays search plus working-memory
to bind them. A diagram is indexed by location: put the premises of an
inference at one spot and the conclusion becomes a *perceptual* event —
an alignment, a containment, a gap — that the visual system computes for
free. Drawing even performs inference: draw two parallel lines cut by a
transversal and eight angle equalities exist on the page unasked.

So a figure's value is never its information content. It is the cost
profile it induces: which inferences became perception. The corollary is
the deepest test we have, and it subsumes most rules below:

> **The computational-advantage test.** Name the specific inference the
> reader must make. Does the figure make it perceptual — premises
> co-located, conclusion readable as a visual feature? If the reader
> still has to compute symbolically after looking, the figure is the
> prose redrawn, and fails.

Two more mechanisms set the ambition. Thurston: mathematicians think in
rich mental imagery but "have fewer and poorer figures in their papers
than in their heads" — inside a subfield a result transfers in minutes
with gestures and pictures, on paper it takes twenty symbolic pages. The
figure's job is to be that wide channel in print: **reify the expert's
private picture**, the one they use but never publish, not an
illustration of the text. And Needham's Feynman epigraph: logically
equivalent representations are psychologically inequivalent — "different
views suggest different kinds of modifications." The representation you
install determines what the reader can *do next*, which is the actual
goal (Nielsen: replace habits of thought, don't deliver facts).

## What deserves a figure

- **No claim, no figure** (Doumont). The figure argues a one-sentence
  claim, not a topic. "Overview of the system" is not a claim; without
  one, every downstream choice (what to include, what to emphasize) is
  undecidable and defaults to decoration. This is why CLAIM is step 1.
- **Draw only what words cannot construct** (Needham's practice). He
  spends no figure on the microscope-lens narration or the dog-on-a-leash
  statement of Rouché — when a physical sentence builds the image
  reliably, prose is *preferred*, because it makes the reader run their
  own visual machinery. Before drawing, try to write the sentence that
  makes the drawing unnecessary. Draw only if the sentence fails —
  i.e. the configuration has specific geometry the reader would get
  wrong.
- **Structure must match** (Larkin & Simon's "sometimes"). The
  diagram advantage exists only when the content has relational
  structure the plane can encode — topology, geometry, dataflow, a state
  space, order-by-two-attributes. A linear chain of implications is
  already correctly indexed as a sequence; a definition is a sentence.
  Figuring those adds translation cost.
- **Division of labor** (Nelsen, Thurston). The figure carries *why* —
  mechanism, gestalt; the equation carries the exact relation; prose
  carries the argument and its qualifications. Proofs-without-words
  "are not really proofs": don't ask the figure to do the equation's
  job, and don't let the equation pretend to do the figure's.
- **Two openings Needham uses that our claim-first habit would miss:**
  the **phenomenon-figure** — open with an unexplained observable the
  reader's current theory cannot account for (z² preserving little
  squares, captioned "A Puzzling Phenomenon"), creating a debt the
  argument then repays; and the **counterexample before the
  definition** — draw a generic member of the unrestricted class
  misbehaving (the "NOT analytic" fan), so the definition arrives as
  the exclusion of a picture the reader has seen. Both are still
  claim-driven; the claim is just delivered as a provocation. Nielsen's
  point: surprise-with-resolution is the one emotion available to a
  static figure, and it outperforms recording.

## Choosing the representation

- **Choose the primitive on which the hypothesis is checkable by eye.**
  Needham is explicit that this is a proof decision: "the use of
  triangles … was not incidental, but instead crucial. Rectangles …
  would simply not have sufficed." Triangles because similarity is
  angle-determined; infinitesimal circles because they are the
  fingerprint of conformality; the polar grid because z² respects it.
  Ask: which geometric primitive makes the theorem's hypothesis a
  visible property? Build the figure from that primitive.
- **Make a ∀ visible as a population.** For "the map does the same to
  everything," draw a *diverse sample* treated uniformly — Needham's
  circle + triangle + blob + parallelogram all rotated-and-expanded at
  once. The figure proves the quantifier by exhibiting the invariant
  across visibly different instances. Omit whatever would single out
  one instance (coordinates, values).
- **One frame of the film.** For a limiting or deformation argument,
  draw the strategically chosen still — usually the nearly-degenerate
  frame where the conclusion first becomes obvious — and let prose
  carry the continuity. The hypothesis that must survive the process
  ("the origin is never crossed") is checked by inspection of the
  still. The reader runs the movie; that's their job, not the ink's.
- **Change representation when the picture saturates.** When a figure
  stops making the next step obvious, don't decorate it — re-plot the
  same object in coordinates where the step is trivial (Needham's
  Hopf proof: loop in plane → standardized loop → graph of Φ(θ), where
  the homotopy becomes linear interpolation between graphs). Each
  representation is drawn only for the property it makes trivial.
- **Fix the abstraction rung, and anchor it** (Victor). Decide
  explicitly: concrete instance / abstracted over time (trajectory) /
  over a parameter (family) / over the family (behavior map). One rung
  per figure. When two rungs are needed, draw the correspondence (the
  inset instance in the abstract map, the accented trajectory in the
  family) — the transition between rungs is where the insight lives,
  so it must be visible, not implicit. And never present an abstraction
  whose lower rungs were never built: a schematic with only Greek
  letters installs no mental model. Carry one fully-bound concrete case
  inside it.
- **Use idioms the reader owns** (Larkin & Simon, Thurston). Diagrams
  are useful only to readers who have the perceptual productions to
  exploit them; a field's visual idiom is "not alive except to those
  who use it." Deviate from the standard visual convention only when
  the deviation is the point; if a needed idiom isn't owned, the figure
  must teach its own reading or be cut.

## Designing the read

A static figure is consumed serially: attention enters, traverses,
exits, and adjacency controls the path. Script it before drawing.

- **The two-clock test** (Tufte's macro/micro made operational). A
  3-second glance must yield the claim; 30 seconds must yield the
  mechanism. If the glance read is empty or wrong, the figure fails
  regardless of its detail. Detail is added *under* the macro structure
  (layering, weight), never beside it — "to clarify, add detail" only
  works when the detail is subordinated.
- **Locality is the layout objective.** Everything used together in one
  inference sits at one location; the cue for the next inference sits
  adjacent. Layout is compiled control flow. Multi-panel figures whose
  panels must be mentally superimposed are prose wearing a diagram's
  clothes.
- **Contiguity is non-negotiable** (Mayer, d ≈ 1.10 — the largest
  measured effect in the literature). Every label sits on the element
  it names. Legends, symbol keys, and caption indirection are
  quantified damage. (grammar.md's "attach labels to what they name" is
  this rule; here is its evidence.)
- **Delete, then signal** (Mayer: coherence d ≈ 0.86, signaling
  d ≈ 0.46 — in that order). First remove everything off-message,
  including true and interesting content; then cue the argument's
  skeleton (weight, accent, an arrow). Signaling on top of clutter is
  noise about noise. Deletion is by message-relevance, never by
  ink-count — Tufte's added detail is on-message at finer grain.
- **Put the message on position** (Cleveland & McGill). Perceptual
  accuracy ranks: position on a common scale > position on nonaligned
  scales > length > angle/slope > area > volume > color shading. The
  quantity carrying the claim gets the top of the hierarchy; encoding
  it as area or saturation squanders the message.
- **Audit the free inferences.** A diagram asserts whatever position,
  length, and slope happen to say, encoded or not — layout accidents
  become claims, truncated axes become lies. This is how correct
  figures mislead. The honesty pass covers deliberate lies; this
  covers accidental ones.

## Reader effort

The literatures conflict here — cognitive load theory wants zero reader
work, proofs-without-words and the self-explanation effect reward
withheld conclusions — and the resolution is a budget with two accounts:

> **Spend nothing on decoding, spend deliberately on inference.** Any
> effort spent figuring out what axes, arrows, or colors mean is pure
> loss. Effort spent completing the punchline inference — one
> perceptual step the figure sets up but doesn't state — is how the
> reader comes to own the mechanism.

Needham's practice is the calibration standard: his delegated steps
("[exercise]", "verify the values given in the figure", "trace it out
with your finger") are always *verification against an answer the
figure already contains* or a bounded failure analysis — never the
creative leap, which he keeps. Stepping stones "sufficiently far apart
that you may need to pause and stretch slightly." The figures are built
to be executed, not viewed.

Two more of his effort devices worth reusing:

- **Name the pictured action.** "Amplitwist" exists so later prose can
  point at a figure's operation with one word; at the limit the figure
  enters the equation itself (f′(z) · ↓ = ↗). When a figure's action
  will recur, coin the verb.
- **Scaffold the unfamiliar with a redrawn familiar.** Before the
  complex derivative, redraw the *real* derivative with split axes so
  the new figure differs from an owned one in exactly one respect —
  the content is the diff. Then state where the analogy breaks.

## Failure modes (correct, pretty, and worthless)

1. **The prose redrawn.** Informationally equivalent to the text, same
   cost structure, no inference got cheaper. The standard ML
   architecture diagram. Diagnostic: is there any question the reader
   answers faster from the figure than from the prose?
2. **Non-local layout.** The key inference's premises live in different
   panels; the reader shuttles, re-importing the working-memory load
   the figure existed to remove.
3. **False free inferences.** Unencoded visual accidents read as
   claims (see audit above).
4. **No macro read.** Works only if fully studied; reviewers don't
   fully study.
5. **Ungrounded abstraction.** The all-symbols schematic (no instance),
   or its twin, the anecdote (no abstraction).
6. **No message.** Caption says "Overview of…"; emphasis undecidable;
   decoration results.
7. **Aesthetic substitution.** Polish raises the probability of being
   read; only computational advantage determines whether reading it
   does anything. Our theme layer makes this failure *easier* — a RISO
   figure looks finished before it argues anything. The gates exist
   because of this.

## Where this binds to the pipeline

- **Design step** (architecture.md): steps 0–3 now carry the selection
  test, representation choice, and traversal script.
- **Readback gate**: the cold reader now reports a glance read first —
  the 3-second claim — before the studied read. A figure whose glance
  read misses is a macro-structure failure even if the studied read
  recovers.
- **Judge**: the comparative judge's standing question gains a sharper
  form — not "which is better drawn" but "in which figure did more of
  the argument become perception?"
