"""derivation_row: the equation IS the figure.

Terms are laid out left-to-right at their real typeset widths; operators
sit midway in the whitespace between neighbors; glosses hang beneath their
terms in the sans register. Everything positional is asserted against the
same metrics the mechanical gate boxes with (typeset.render_math).
"""

import math

import pytest

from figlib.derivation import Term, derivation_row
from figlib.scene import Curve, MathLabel
from figlib.style import Role
from figlib.typeset import PX_PER_PT, render_math

PXU = 40.0          # px per math unit for these tests
SIZE_PT = 11.0      # the builder's default reading size

TERMS = [
    Term(r"\tfrac{1}{2}mv^2", gloss="kinetic"),
    Term(r"mgh", gloss="potential"),
    Term(r"W"),
]


def term_width_units(latex: str, size_pt: float = SIZE_PT) -> float:
    return render_math(latex, size_pt).width_px / PXU


def term_height_units(latex: str, size_pt: float = SIZE_PT) -> float:
    return render_math(latex, size_pt).height_px / PXU


def row(terms=TERMS, **kw):
    kw.setdefault("px_per_unit", PXU)
    return derivation_row(terms, **kw)


def labels_of(items):
    return [it for it in items if isinstance(it, MathLabel)]


def term_labels(items, terms=TERMS):
    wanted = {t.latex for t in terms}
    return [it for it in labels_of(items) if it.latex in wanted]


def gloss_labels(items):
    return [it for it in labels_of(items) if it.register == "sans"]


class TestLayout:
    def test_anchor_count_matches_terms(self):
        _, anchors = row()
        assert len(anchors) == len(TERMS)

    def test_anchors_on_the_row_line(self):
        _, anchors = row(y=2.5)
        assert all(a[1] == pytest.approx(2.5) for a in anchors)

    def test_strict_left_to_right_non_overlap(self):
        _, anchors = row()
        widths = [term_width_units(t.latex) for t in TERMS]
        for i in range(len(TERMS) - 1):
            dx = anchors[i + 1][0] - anchors[i][0]
            assert dx >= (widths[i] + widths[i + 1]) / 2.0 - 1e-9, (
                f"terms {i} and {i + 1} overlap: centers {dx:.4f} apart, "
                f"half-widths sum {(widths[i] + widths[i + 1]) / 2.0:.4f}")

    def test_ops_midway_between_neighbor_edges(self):
        items, anchors = row(op="+")
        ops = [it for it in labels_of(items) if it.latex == "+"]
        assert len(ops) == len(TERMS) - 1
        widths = [term_width_units(t.latex) for t in TERMS]
        ops = sorted(ops, key=lambda o: o.anchor[0])
        for i, o in enumerate(ops):
            left_edge = anchors[i][0] + widths[i] / 2.0
            right_edge = anchors[i + 1][0] - widths[i + 1] / 2.0
            assert o.anchor[0] - left_edge == pytest.approx(
                right_edge - o.anchor[0], abs=1e-9)

    def test_gap_pt_is_a_floor_around_ops(self):
        gap_pt = 14.0
        items, anchors = row(gap_pt=gap_pt)
        ops = sorted((it for it in labels_of(items) if it.latex == "+"),
                     key=lambda o: o.anchor[0])
        widths = [term_width_units(t.latex) for t in TERMS]
        op_half = term_width_units("+") / 2.0
        gap_units = gap_pt * PX_PER_PT / PXU
        for i, o in enumerate(ops):
            left_edge = anchors[i][0] + widths[i] / 2.0
            clear = (o.anchor[0] - op_half) - left_edge
            assert clear >= gap_units - 1e-9

    def test_lhs_prepended_with_equals(self):
        items, anchors = row(lhs="E")
        lhs = [it for it in labels_of(items) if it.latex == "E"]
        eq = [it for it in labels_of(items) if it.latex == "="]
        assert len(lhs) == 1 and len(eq) == 1
        assert lhs[0].anchor[0] < eq[0].anchor[0] < anchors[0][0]
        assert len(anchors) == len(TERMS)  # anchors are TERM anchors only


class TestGlosses:
    def test_gloss_below_its_term_in_sans(self):
        items, anchors = row()
        glosses = gloss_labels(items)
        assert len(glosses) == 2  # two glossed terms
        for i, t in enumerate(TERMS):
            if t.gloss is None:
                continue
            g = next(g for g in glosses
                     if math.isclose(g.anchor[0], anchors[i][0], abs_tol=1e-9))
            assert g.latex == t.gloss
            assert g.role is Role.ANNOTATION
            assert g.anchor[1] < anchors[i][1]

    def test_default_gloss_drop_is_3_term_heights(self):
        # Clearance needs > 2.5 term heights or the tick runs through the
        # term's own glyphs (verified on the qk figure); 3.0 is the default.
        items, anchors = row(y=0.0)
        glosses = sorted(gloss_labels(items), key=lambda g: g.anchor[0])
        assert glosses, "no glosses produced"
        for g, (i, t) in zip(glosses,
                             [(i, t) for i, t in enumerate(TERMS) if t.gloss]):
            h = term_height_units(t.latex)
            assert g.anchor[1] == pytest.approx(-3.0 * h)

    def test_gloss_tick_curves(self):
        items, anchors = row()
        ticks = [it for it in items if isinstance(it, Curve)]
        assert len(ticks) == 2
        for tk in ticks:
            assert tk.role is Role.ANNOTATION
            xs = tk.pts[:, 0]
            assert float(xs.max() - xs.min()) == pytest.approx(0.0, abs=1e-9)
        # each tick is vertical, ~0.6 of the drop, aligned with a glossed term
        glossed_x = {anchors[i][0] for i, t in enumerate(TERMS) if t.gloss}
        for tk in ticks:
            x = float(tk.pts[0, 0])
            assert any(math.isclose(x, gx, abs_tol=1e-9) for gx in glossed_x)
            span = float(tk.pts[:, 1].max() - tk.pts[:, 1].min())
            i = next(i for i, t in enumerate(TERMS)
                     if t.gloss and math.isclose(anchors[i][0], x, abs_tol=1e-9))
            drop = 3.0 * term_height_units(TERMS[i].latex)
            assert span == pytest.approx(0.6 * drop, rel=1e-6)


class TestDiscipline:
    def test_every_label_pinned_and_centered(self):
        items, _ = row(lhs="E")
        labs = labels_of(items)
        assert labs, "no labels produced"
        for lab in labs:
            assert lab.pin, f"{lab.latex!r} not pinned: position IS meaning"
            assert lab.ha == "center" and lab.va == "center"

    def test_term_role_and_key_carried(self):
        terms = [Term("a", role=Role.ACCENT1, key="k1"), Term("b")]
        items, _ = row(terms)
        a = next(it for it in labels_of(items) if it.latex == "a")
        assert a.role is Role.ACCENT1 and a.key == "k1"


class TestReadingPxPerUnit:
    def test_reading_px_per_unit_divides_out_ink_scale(self):
        # px_per_unit here is READING px per math unit. At WIDE the render
        # multiplies every px quantity by ink_scale = 1.45, so the canvas
        # scale (schematic.px_per_unit) is 1.45x too tight for typeset
        # metrics measured at reading size.
        from figlib.derivation import reading_px_per_unit
        from figlib.format import WIDE, COLUMN

        xlim = (0.0, 500.0)
        pad = 0.08
        got = reading_px_per_unit(WIDE, xlim)
        want = WIDE.display_width_px / (WIDE.ink_scale * 500.0 * (1 + 2 * pad))
        assert got == pytest.approx(want)
        # at ink_scale 1 it coincides with the canvas-px bridge
        from figlib.schematic import px_per_unit
        assert reading_px_per_unit(COLUMN, xlim) == pytest.approx(
            px_per_unit(xlim, COLUMN.display_width_px))
