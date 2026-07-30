# Readback record

**Verdict:** pass

## Intended claim

Least-squares polynomial fits of one 10-point sample of sin(2*pi*x) + noise go from underfitting to interpolating as the degree M rises: M = 0 and M = 1 cannot reach the signal, M = 3 recovers it, and M = 9 — one free coefficient per data point — passes exactly through every observation and swings off the frame between them, fitting the noise rather than the function.

## Cold readback

Cold agent (no CLAIM, no source, no conversation context), prompt from
`figcheck --readback-prompt`, on the shipped PNG.

**1. GLANCE.** Polynomial curve fitting to noisy samples of a sine — four
panels showing a red fitted curve getting progressively wigglier as the
degree M rises, asserting that too low a degree underfits and too high a
degree overfits.

**2. STUDY — the claim.** Fitting a degree-M polynomial by least squares to
N = 10 noisy observations of sin 2πx recovers the underlying function well
only at intermediate M (M = 3): M = 0 and M = 1 are too rigid to represent
the sinusoid, while M = 9 — with as many free coefficients as data points —
interpolates every observation exactly and thereby tracks the noise instead
of the signal, oscillating wildly between and beyond the data.

**3. Confusions.**
- The M = 9 curve visibly runs off the top and bottom of the plotting
  region and the t-axis has no drawn bound or tick beyond ±1, so it is not
  clear whether the clipping is a deliberate "it blows up" statement or the
  curve simply exceeding an unstated axis range. In [d] the red curve is
  broken into disconnected strokes where it leaves the frame, which reads
  as a rendering artifact rather than a mathematical feature.
- No residual or error number is shown anywhere — no RMS, no train-vs-test
  curve — so "M = 3 is best" is asserted only by visual proximity of red to
  teal, not quantified.
- The noise model (Gaussian? what σ?) is never given, and two dots sit
  noticeably above +1.4, well outside the sine's range, which I had to
  assume is intended noise rather than an outlier being made a point of.
- The degree sequence 0, 1, 3, 9 skips 2 and everything between 4 and 8; I
  had to assume these are representative rather than the only interesting
  cases.

**4. Verified by inspection vs. taken on trust.** Verifiable: that the same
10 data points and the same teal sin 2πx appear in all four panels; that
the M = 0 fit is a horizontal line near the data mean; that M = 1 is a
straight line with negative slope; that M = 3 stays close to the teal curve
across the whole interval; that M = 9 passes through (or very near) every
plotted dot while departing far from the teal curve between them; that the
degree/wiggle relationship is monotone in the direction claimed. On trust:
that each red curve is genuinely the least-squares minimizer for its degree
(only plausibility is checkable — the M = 0 line looks like the mean, a
weak check); that the data really were generated from sin 2πx plus noise
rather than fitted post hoc; that M = 9 has exactly zero training error
(the figure draws it that way but never says it); and the entire
generalization claim — nothing here shows held-out data, so "M = 9 is
worse" is an inference from the drawing's premise that teal is the truth.

**5. BINDING — held the same.** The 10 navy observation dots (identical
x-positions and identical t-values in every panel — same sample, not four
independent draws); the teal sin 2πx curve; the axis frame, the 0-to-1
x-range, the −1/0/1 t-ticks, the axis labels t and x; the "N = 10
observations" annotation; the red colour and the label template "least
squares polynomial, degree M = ·"; panel size and layout.

**6. Differences.** The stated degree (0, 1, 3, 9); the shape of the red
curve (constant → sloped line → smooth cubic tracking the sine → high
-frequency oscillation); the perceived red stroke weight, which looks
heavier in [c] and [d] — "I'm not certain this is intentional"; only in [d]
does the red curve leave the plotting area and get clipped/broken; the
vertical extent of red ink; how close red comes to the dots (ignores them
in [a]/[b], averages through them in [c], passes through essentially all of
them in [d]); where the red curve starts and ends ([c] and [d] appear to
terminate on the first and last dots); the panel labels [a]–[d].

**7. The single difference argued about.** The polynomial degree M — model
complexity — specifically what happens as M goes from too small (0, 1)
through about right (3) to as-many-parameters-as-data (9). "Everything else
is deliberately held fixed so that M is the only free variable."


## Notes

Protocol: context-free subagent on the PNG only, prompt verbatim from
`figcheck --readback-prompt` including the 4-part binding questions. Not a
self-read; the reader had no access to this program, the CLAIM, or the
conversation. Five such reads were run over the iteration; the one recorded
above is on the shipped render.

**Verdict: pass.** The GLANCE read names capacity as the variable and
"underfit / overfit" as the assertion — the macro-structure lands before
any study. The STUDY read reproduces CLAIM almost clause for clause,
including the load-bearing mechanism (as many free coefficients as data
points -> exact interpolation -> fitting the noise). Answer 7 recovers the
declared `Correspondence` exactly: `varies` = the degree M, and the reader
volunteered that everything else is held fixed *on purpose*. The binding
answers and the residual gate agree, which is the point of having both.

**Fixed in response to earlier reads:**
1. *"M is never defined; nothing says the red curve is a least-squares
   fit."* Three readers in a row inferred "polynomial degree" correctly
   from the shapes and each flagged that they were inferring it. The panel
   caption became "least squares polynomial, degree M = k" — the estimator
   and what M indexes are exactly the two things this figure argues you
   should not take on faith.
2. *"The dots are never labelled; N is never stated, so I had to count."*
   Added "N = 10 observations" in the data's own ink, in the empty strip
   below the signal. N belongs with the dots, not in the caption: M = 9 =
   N − 1 is the punchline, and a reader made to count is doing the figure's
   work.
3. *"In [d] the fit crosses the x-axis arrow region."* The x spine moved
   from 0.12 to 0.24 below the truncation band.
4. *"The sin 2πx label collides with the M = 9 spike."* Re-anchored to
   x = 0.40, where the maximum of all four fits is 0.55 — computed, not
   nudged, so auto-place leaves it alone and the label lands on the same
   pixel in all four panels.

**Accepted, not fixed:**
- *The truncation is not quantified.* The M = 9 fit reaches −2.83 and the
  frame stops at −1.5. Every reader did read "the curve leaves the plotted
  range", which is the half that argues; the number is a caption fact. A
  dotted FRAME rule at t = ±1.5 was built and cut: at FRAME weight the
  reader looking at that render still asked whether the break was a
  clipping artifact and never mentioned the rules, and at any heavier
  weight they read as a second pair of content lines.
- *No error number, no held-out data.* Correct, and deliberate — the
  train/test curve is PRML Fig 1.5, a different figure with a different
  claim. This one is the sweep.
- *The noise model is not on the page.* One seed, one draw, σ = 0.25, named
  in PARAMS and admitted in the docstring's honesty pass.

**False positives worth recording** (the residual gate proves all four
panels identical outside `fit`, so these are perceptual, not real):
- Two readers reported the panels are "not pixel-identical in framing" —
  one saw the right column shifted, one saw the [b] axis higher. The grid
  is equal slots and the panels share one `Scene` frame, asserted.
- Two readers reported the red stroke weight varying across panels. It is
  one `width_scale` for all four; the perception tracks curvature and ink
  density, not stroke width. Worth knowing that a "family drawn at one
  weight" can still read as varying weight when the arclength differs by
  4x.

