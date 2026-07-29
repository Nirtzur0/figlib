# Friction records

One record per figure built or rebuilt, written by the agent that did the work,
at the time it did the work — otherwise the knowledge of which primitive was
missing dies with the session.

The synthesis is mechanical over the structured fields below: rank hand-rolled
devices by (figures affected × lines hand-rolled), fold the ranking into
`primitive-gaps.md`, build the top few. So fill the fields in literally.

## What the first 12 records already showed

Worth knowing before you write yours, because it is the pattern:

- **When the gate contains the fix, a figure is cheap. When it doesn't, it
  costs ~3×.** The cheapest figure in the first batch hit green in 3 renders,
  both failures being collisions the mechanical gate pinpointed exactly. The
  two most expensive needed a *design* change no diagnostic would ever
  prompt — and cost 200k+ tokens each.
- **The dangerous failures are the ones that pass.** Three independent agents
  hit "gates green, picture wrong": a marker size passed in canvas px instead
  of scene units produced a 100%-ink solid canvas that passed every gate; a
  hollow `Point`'s opaque fill silently erased the dot beneath it; the
  ink-free search recommended a position outside `clip="frame"`, where a label
  renders invisible. Gates check neither drawn ink area nor occlusion.
- **Two agents independently burned effort on `correspond.py`** before both
  abandoned `CORRESPONDENCE` — its fingerprint deliberately ignores position,
  so a binding whose whole point is a length change cannot be expressed. The
  same dead end, paid for twice, because nothing documented it.

## Template

Copy into `docs/friction/<name>.md`.

```markdown
# <figure> — friction record        agent · subject · date

## What the claim needed
One paragraph: the geometry and annotation the CLAIM demanded.

## What figlib gave for free
What worked without a fight. This protects working machinery from being
"improved" later — say what you leaned on.

## What I hand-rolled
The load-bearing section. One bullet per device, with a line count:

- <device> — ~N lines in the figure program that should be a primitive

## Gate diagnostics that did NOT contain the fix
`CLAUDE.md` claims diagnostics contain the fix. Where that failed, quote the
diagnostic and say what you actually had to do. "None" is a real result.

## Anything that passed the gates but was wrong
The most valuable field. What did you catch only by looking at the PNG?

## Renders to first green
N

Count every `make check` from first run to first PASS. This is what ranks
friction across the corpus, so do not estimate it.

## Proposed primitive
A signature, not prose. "None" is valid — say why the hand-rolling was
inherent to this figure rather than general.
```
