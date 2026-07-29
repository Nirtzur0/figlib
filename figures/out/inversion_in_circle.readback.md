# Readback record

**Verdict:** pass

## Intended claim

Inversion z -> R^2/z-bar sends circles through the center of inversion to lines, circles that miss the center to circles, fixes the circle |z| = R pointwise, and preserves the angle at which two curves cross in magnitude while reversing its sense -- so an orthogonal pair of circles stays orthogonal, a right angle mapping to a right angle of the opposite hand.

## Cold readback

GLANCE: Inversion of a pencil of circles through a point, seen against a
fixed circle |z|=R -- asserting that the map preserves tangency/right
angles between a moving (blue/red) pair of circles while restructuring
the rest of the picture.

STUDY:
2. Claim: inversion z -> R^2/z-bar in the circle |z|=R sends circles
through 0 to lines and circles missing 0 to circles, fixes |z|=R
pointwise, and preserves the orthogonality of the tracked blue/red pair
across the map (with the angle's sense reversed, per the 90+/90- labels).
3. Confusion: (a) which circles pass through 0 has to be inferred from the
two panels together, not read directly off panel [a] alone; (b) the
+/- subscript convention on the 90-degree labels is not self-explanatory
without the caption; (c) the point "R" is not explicitly glossed as "the
fixed point on the inversion circle" versus just a marked sample point.
4. Verifiable by inspection: the dashed |z|=R circle is pixel-identical in
both panels (the fixed locus); the right-angle tick sits unambiguously
between the blue and red curves in both panels, with no other curve
passing near that crossing. Taken on trust: the algebra of z -> R^2/z-bar
itself, and that the color correspondence (blue stays blue, red stays
red) is the intended binding rather than an accident.

BINDING:
5. Same: the dashed |z|=R circle (identical, fixed); the color/role of
the tracked blue and red curves; the right-angle tick glyph; the "R"
point.
6. Different: three generic through-origin circles fan out from 0 in
different directions in [a] and become three lines crossing at different
angles in [b]; the blue circle (through 0) becomes a straight line; the
red circle (missing 0) stays a circle; "0" has no counterpart in [b]
(replaced by the explicit "0 -> infinity" note on the connector); the
angle label switches from 90+ to 90-.
7. The single difference argued: whether a circle passes through the
center of inversion determines circle-to-line vs. circle-to-circle,
while the right angle between the tracked pair survives either way.
8. The right-angle tick sits unambiguously between the blue and red
curves in both panels -- no third curve passes near enough to compete
for the reading, in either panel.


## Notes

Three cold-reader passes were run during design. Pass 1 flagged: (a) the inversion circle was never labeled, (b) 'R' collided with the fixed point's role, (c) 0 appeared unchanged in the image panel as if fixed. All three were fixed: an explicit |z|=R callout, a '0 -> infinity' note on the connector, and removing the false 0-point from the mapped panel. Pass 2 flagged a genuine geometry defect invisible to any gate: the orthogonal pair's crossing point coincided almost exactly with where the inversion circle and a generic family member also passed, so the reader attributed the right-angle mark to the wrong pair of curves. Fixed by redesigning the through-origin family as a pencil fanning out in different directions from 0 (rather than three circles stacked on one ray) and re-deriving the second orthogonal circle's center by a small grid search that maximizes the crossing point's distance from every other drawn curve. Pass 3 confirms the crossing now reads as unambiguously blue-vs-red in both panels. Remaining confusions (the +/- sign convention, which circles pass through 0) are reader-delegated inferences the figure intends the study-pass to resolve, not defects.
