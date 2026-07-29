# Readback record

**Verdict:** pass

## Intended claim

As z traces Gamma once counterclockwise, f(z) = (z-a)^2 (z-b) traces a closed curve that winds around w = 0 exactly three times — twice for the double zero at a, once for the simple zero at b — because the winding number of f(Gamma) equals the enclosed zero count with multiplicity.

## Cold readback

GLANCE: A closed curve around two marked points on the left maps, under some function, to a curve on the right that loops around the origin three times — this is about how many times an image curve encircles a point.

STUDY:

2. The map f(z) = (z−a)²(z−b) sends a simple closed contour Γ once around a and b to an image curve f(Γ) that winds around the origin exactly 3 times — i.e., the winding number of f(Γ) about 0 equals the total multiplicity of zeros enclosed by Γ (2 from the double zero at a, 1 from the simple zero at b).

3. Confusions/ambiguities:
   - "(double)" label near a is small and could be misread as annotating b instead, since it sits between the two dots.
   - It's not visually obvious from panel [b] alone that the curve crosses itself three separate times rather than some other self-intersection count — had to trace the arrowheads to confirm three concentric-ish loops.
   - The correspondence between the two red/blue dots in [a] and the single black dot (origin) in [b] is implicit — nothing marks that a and b map to 0 explicitly other than the general shape of f.

4. Verifiable by inspection: that Γ is a single simple closed curve traversed once (arrows show consistent counterclockwise direction, no self-crossings in [a]); that f(Γ) has arrows and appears to wind around the center 3 times (countable by following the arrowheads and the labeled n=3). Taken on trust: that f(Γ) is actually the correct image of Γ under the stated formula (no grid/domain coloring lets the reader check pointwise correspondence), and that a is truly a double zero rather than just a label of convenience.

BINDING:

5. Same across parts: the enclosed interior point(s)/zero location(s) conceptually correspond to the single origin point in [b] — "a and b" (left) correspond to "the point being wound around" (right, the origin). The curve itself (Γ vs f(Γ)) is the same object before/after the map, drawn with matching black stroke and arrowhead style.

6. Differences noticed:
   - Shape: left is a single simple circle; right is a curve with multiple (three) overlapping loops.
   - Number of marked interior points: left has two distinct colored points (a, b); right has one point (origin, unlabeled with a letter).
   - Color: a is blue, b is red, on the left; the right panel's center dot is plain black/uncolored.
   - Annotation style: left uses colored dot + letter labels; right uses an arrow pointing to the label "n = 3".
   - Complexity/self-intersection: left curve has zero self-crossings, right has visible self-crossings (it's not embedded).
   - The "(double)" annotation appears only on the left, with no analog on the right.

7. The figure is arguing about winding number / multiplicity: specifically that the image curve's winding count around the origin (n = 3) equals the sum of the zero multiplicities enclosed by the original contour (2 for a's double zero + 1 for b's simple zero).

## Notes

Both GLANCE and STUDY recovered the intended claim exactly, including the
2 + 1 = 3 multiplicity breakdown — no macro-structure failure and no
mathematical misread. Two items are accepted as-is rather than fixed:

- The reader had to *count* self-crossings in panel [b] rather than read
  the count directly. This is honest: the winding number is genuinely a
  counting fact about the curve, and the drawn `n = 3` callout plus the
  arrow ticks are exactly the affordances that make the count checkable
  rather than asserted. No further encoding (e.g. per-loop coloring) was
  added, since it would tint what the reader is supposed to count.
- The reader flagged that nothing marks "a and b map to 0" explicitly.
  This is deliberate scope: this figure's claim is the winding COUNT, not
  the pointwise map; a per-point correspondence (e.g. a dashed construction
  line from a/b to the origin) would be a different, denser figure. Not
  drawn, and no CORRESPONDENCE is declared for exactly this reason — the
  two panels are not a tracked-object binding, they are premise and
  consequence of one counting argument.

The "(double)" label placement ambiguity (bullet 1 of Q3) is a real,
cheap fix candidate (nudge it closer to `a` or bold the letter), but does
not change the verdict since the reader still resolved it correctly by
the STUDY pass.
