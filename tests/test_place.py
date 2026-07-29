"""Placement along an author-chosen locus.

The transcripts' most expensive repeated move: a label whose position is a
PARAMETER of the geometry (a radius and an angle on a ring, a fraction
along a curve) gets hand-searched over several render cycles — 1.17, 1.32,
1.40, 1.52 — with the clearance arithmetic done by hand in reasoning.
`autoplace` cannot help: it nudges `offset_px` on un-pinned labels, and
text along a curve is pinned by construction.

What is under test is that the 1-D search is now DERIVED and reported: the
author still chooses the locus (that is the meaning-bearing decision), the
library picks the point on it, and the achieved clearance comes back as a
number the program can assert on instead of a number someone eyeballed.
"""

import numpy as np
import pytest

from figlib.place import label_clearance, place_on_locus
from figlib.scene import Curve, MathLabel, Scene
from figlib.style import DEFAULT_STYLE, Role

W = 600.0


def _scene_with_a_bar():
    """A horizontal content stroke across the middle of the canvas."""
    s = Scene(xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
    s.add(Curve(np.array([[-1.0, 0.0], [1.0, 0.0]]), role=Role.CONTENT))
    return s


class TestPlaceOnLocus:
    def test_picks_the_candidate_farthest_from_content_ink(self):
        s = _scene_with_a_bar()
        # a vertical locus crossing the bar: the ends are clear, the middle is not
        locus = [(0.0, y) for y in np.linspace(-0.9, 0.9, 19)]

        p = place_on_locus(s, DEFAULT_STYLE, r"x", locus, width_px=W)

        assert abs(p.anchor[1]) > 0.5, "landed near the ink it should avoid"
        assert p.clearance_px > 0.0
        assert p.index in (0, len(locus) - 1)

    def test_a_single_candidate_locus_returns_that_candidate(self):
        s = _scene_with_a_bar()

        p = place_on_locus(s, DEFAULT_STYLE, r"x", [(0.4, 0.7)], width_px=W)

        assert p.index == 0
        assert p.anchor == pytest.approx((0.4, 0.7))

    def test_existing_labels_are_obstacles_too(self):
        s = Scene(xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
        s.add(MathLabel(r"\text{TAKEN}", (0.0, 0.8), ha="center", va="center"))
        locus = [(0.0, 0.8), (0.0, -0.8)]

        p = place_on_locus(s, DEFAULT_STYLE, r"\text{NEW}", locus, width_px=W)

        assert p.index == 1, "sat on top of an existing label"

    def test_clearance_is_reported_in_canvas_px_and_is_monotone(self):
        s = _scene_with_a_bar()
        near = place_on_locus(s, DEFAULT_STYLE, r"x", [(0.0, 0.05)], width_px=W)
        far = place_on_locus(s, DEFAULT_STYLE, r"x", [(0.0, 0.9)], width_px=W)

        assert far.clearance_px > near.clearance_px

    def test_ties_break_to_the_lowest_index_so_the_result_is_deterministic(self):
        s = Scene(xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))   # empty: every spot ties
        locus = [(0.0, 0.5), (0.0, -0.5)]

        a = place_on_locus(s, DEFAULT_STYLE, r"x", locus, width_px=W)
        b = place_on_locus(s, DEFAULT_STYLE, r"x", locus, width_px=W)

        assert a.index == b.index == 0
        assert a.anchor == b.anchor

    def test_tangent_angle_follows_the_locus_on_the_page(self):
        s = Scene(xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
        # a locus running up-and-right at 45 degrees in math coords
        locus = [(t, t) for t in np.linspace(-0.5, 0.5, 11)]

        p = place_on_locus(s, DEFAULT_STYLE, r"x", locus, width_px=W,
                           angle_from_tangent=True)

        assert p.angle_deg == pytest.approx(45.0, abs=1.0)

    def test_tangent_angle_stays_readable_never_upside_down(self):
        s = Scene(xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
        locus = [(-t, -t) for t in np.linspace(-0.5, 0.5, 11)]   # runs down-left

        p = place_on_locus(s, DEFAULT_STYLE, r"x", locus, width_px=W,
                           angle_from_tangent=True)

        assert -90.0 < p.angle_deg <= 90.0

    def test_the_locus_is_the_authors_choice_and_is_never_left(self):
        s = _scene_with_a_bar()
        locus = [(0.0, y) for y in np.linspace(-0.9, 0.9, 19)]

        p = place_on_locus(s, DEFAULT_STYLE, r"x", locus, width_px=W)

        assert tuple(p.anchor) in {tuple(q) for q in locus}


class TestLabelClearance:
    def test_measures_an_existing_labels_room(self):
        """The query that replaces 'distance works out to about 0.108 units'
        done by hand in a reasoning block."""
        s = _scene_with_a_bar()
        far = MathLabel(r"x", (0.0, 0.9), ha="center", va="center")
        near = MathLabel(r"y", (0.0, 0.02), ha="center", va="center")
        s.add(far, near)

        assert label_clearance(s, DEFAULT_STYLE, far, width_px=W) > \
            label_clearance(s, DEFAULT_STYLE, near, width_px=W)

    def test_a_label_sitting_on_ink_reports_non_positive_clearance(self):
        s = _scene_with_a_bar()
        on = MathLabel(r"x", (0.0, 0.0), ha="center", va="center")
        s.add(on)

        assert label_clearance(s, DEFAULT_STYLE, on, width_px=W) <= 0.0
