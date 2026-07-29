# Readback record

**Verdict:** pass

## Intended claim

    "Stereographic projection from the north pole N sends a point p of the "
    "plane outside the unit circle to the point p-hat where the ray N->p "
    "pierces the northern hemisphere of the unit sphere; the equator is the "
    "unit circle itself and its points are fixed — the line through p "
    "crosses the unit circle at two points and its image circle passes "
    "through the same two points; a straight line through p maps to a "
    "circle through N, so p -> infinity drives p-hat -> N."

## Cold readback

A tan unit sphere Sigma, poles N (top) and S (open muted dot behind the
southern cap), sits centered at O astride an opaque sheet C drawn through
its equator; the equator is the blue curve labeled UNIT CIRCLE in
perspective lettering on the sheet. A straight black line in the sheet,
hollow-arrowed in both directions, crosses the blue circle at two small
filled black points. Red rods run from N through two red points of the
line (p labeled, a second unlabeled sample); each rod is solid outside
the sphere, dashed through its interior, and pierces the upper hemisphere
at a red image point (p-hat labeled). A black circle through N carries
the same hollow arrows and passes through the SAME two black crossing
points; between them it dips below the sheet and is dashed. Reading:
the line maps to the circle through N; the two unit-circle crossings are
shared by line and image, so equator points stay put; chasing the arrows
outward along the line chases the image point up into N.

## Defect dispositions (final fix round)

- **label-on-ink — `\text{CIRCLE}` on Curve CONTENT ink** (the sphere's
  silhouette; word center projected 1.175 from origin, limb at 1.0, so the
  word's inner end clipped the limb stroke). FIXED: both perspective words
  moved out along the arc, shared radius 1.40 -> 1.52. Halo rejected — a
  halo would cut the limb. The checker's off-arc relocation (-1.12, -0.83)
  rejected — it breaks the in-plane perspective-lettering device.
- **arrow-on-mark — hollow arrowhead of the t=0.6 direction stub on L
  landed on the CIRCLE label.** FIXED: stub dropped. L keeps one hollow
  arrow per solid run (t=-2.2 on the far run, t=1.5 on the near run),
  which is all the direction story needs.
- Autoplace nudges accepted: N (0,-17), p-hat (+2,0), C (-19,0),
  UNIT (0,-6) px.
- Residual accepted: expressivity reports ink 89.6% of canvas with
  FilledCurve CONTENT at 96% — the sheet and body fills dominate by
  design in a solid-3D scene.

## Notes

figcheck PASS; golden regression MATCH. Cold read recovers every clause
of the claim, including the fixed-crossings argument, from ink alone.
The one inference taken partly on faith: that the dashed southern dip of
the image circle is sheet-occlusion (not sphere-occlusion) — the cap
poking below the near rim is the visual cue that disambiguates it.
