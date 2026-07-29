"""The matrix layer: Block geometry, encoders, the expression row, gates."""

import numpy as np
import pytest

from figlib.gates import Checks
from figlib.matrix import (Block, bands, banded, bounds, causal,
                           check_cell_uniform, check_conformable, check_expr,
                           check_hinton_area, check_no_overlap, diagonal,
                           entries, expr, heat, hinton, lattice, mask,
                           outline, rank1, tri)
from figlib.scene import FilledCurve, MathLabel, Point
from figlib.style import Role


# --- Block geometry ------------------------------------------------------

def test_block_extent_puts_row_zero_at_the_top():
    b = Block(2, 3)                      # origin (0,0) is the TOP-LEFT
    assert b.width == 3.0
    assert b.height == 2.0
    assert b.extent == (0.0, 3.0, -2.0, 0.0)
    # row 0 sits above row 1 in math coords (+y up)
    assert b.center(0, 0)[1] > b.center(1, 0)[1]


def test_block_center_and_rect():
    b = Block(2, 3, origin=(10.0, 5.0), cell=2.0)
    assert b.center(0, 0) == (11.0, 4.0)
    assert b.center(1, 2) == (15.0, 2.0)
    r = b.rect()
    assert r.shape == (4, 2)
    assert r[:, 0].min() == 10.0 and r[:, 0].max() == 16.0
    assert r[:, 1].min() == 1.0 and r[:, 1].max() == 5.0


def test_block_cols_and_rows_span_the_full_cross_dimension():
    b = Block(2, 3)
    c = b.cols(1)
    assert c[:, 0].min() == 1.0 and c[:, 0].max() == 2.0
    assert c[:, 1].min() == -2.0 and c[:, 1].max() == 0.0
    r = b.rows(0)
    assert r[:, 0].min() == 0.0 and r[:, 0].max() == 3.0
    assert r[:, 1].min() == -1.0 and r[:, 1].max() == 0.0
    c2 = b.cols(0, 2)
    assert c2[:, 0].min() == 0.0 and c2[:, 0].max() == 2.0


def test_block_sub_composes_and_carries_values():
    V = np.arange(12, dtype=float).reshape(3, 4)
    b = Block(3, 4, values=V, name="A")
    s = b.sub(slice(1, 3), slice(2, 4))
    assert (s.m, s.n) == (2, 2)
    assert s.origin == (2.0, -1.0)
    assert np.array_equal(s.values, V[1:3, 2:4])
    assert s.center(0, 0) == b.center(1, 2)


def test_block_rejects_a_values_shape_mismatch():
    with pytest.raises(ValueError, match="values shape"):
        Block(2, 3, values=np.zeros((3, 2)))


def test_block_rejects_a_nonpositive_shape():
    with pytest.raises(ValueError, match="positive"):
        Block(0, 3)


# --- structure encoders --------------------------------------------------

def test_outline_emits_a_closed_rim_and_an_optional_wash():
    b = Block(2, 3)
    assert len(outline(b)) == 1
    items = outline(b, wash=0.3)
    assert len(items) == 2
    assert isinstance(items[0], FilledCurve)      # wash first, under the rim
    assert items[0].opacity == 0.3
    assert items[1].closed is True


def test_bands_emit_one_fill_per_column_or_row():
    b = Block(2, 3)
    cols = bands(b, "col")
    rows = bands(b, "row")
    assert len(cols) == 3 and len(rows) == 2
    assert all(isinstance(f, FilledCurve) for f in cols)
    c = cols[0].pts
    assert c[:, 0].max() - c[:, 0].min() == 1.0
    assert c[:, 1].max() - c[:, 1].min() == 2.0


def test_bands_gap_insets_each_band_without_moving_its_center():
    b = Block(2, 3)
    c = bands(b, "col", gap=0.2)[1].pts
    assert c[:, 0].min() == pytest.approx(1.1)
    assert c[:, 0].max() == pytest.approx(1.9)


def test_lattice_emits_one_dot_per_entry_at_cell_centers():
    b = Block(2, 3)
    dots = lattice(b)
    assert len(dots) == 6
    assert all(isinstance(d, Point) for d in dots)
    assert dots[0].xy == b.center(0, 0)


def test_mask_merges_consecutive_true_cells_in_a_row():
    b = Block(2, 3)
    M = np.array([[True, True, True], [False, True, False]])
    fills = mask(b, M)
    assert len(fills) == 2
    wide = fills[0].pts
    assert wide[:, 0].max() - wide[:, 0].min() == 3.0


def test_mask_accepts_a_predicate():
    b = Block(2, 2)
    assert len(mask(b, lambda i, j: i == j)) == 2


def test_tri_banded_and_causal_are_masks_over_the_index_grid():
    b = Block(3, 3)
    assert tri(b, "lower").sum() == 6
    assert tri(b, "upper").sum() == 6
    assert banded(b, 0, 0).sum() == 3          # the diagonal
    assert banded(b, -1, 1).sum() == 7         # tridiagonal
    assert np.array_equal(causal(b), tri(b, "lower"))


def test_diagonal_wraps_for_a_circulant_portrait():
    b = Block(4, 4)
    assert diagonal(b, 0).sum() == 4
    assert diagonal(b, 1).sum() == 3           # the superdiagonal is short
    # wrapped, EVERY diagonal of a square block has exactly n cells
    assert diagonal(b, 1, wrap=True).sum() == 4
    assert diagonal(b, -3, wrap=True).sum() == 4
    assert np.argwhere(diagonal(b, 0)).shape == (4, 2)


# --- rank-1 and the value encoders ---------------------------------------

def test_rank1_draws_every_column_with_opacity_tracking_the_coefficient():
    b = Block(3, 3)
    u = np.array([1.0, 2.0, 3.0])
    v = np.array([1.0, 0.5, 0.0])
    fills = rank1(b, u, v)
    # the zero coefficient contributes nothing, so it is not drawn
    assert len(fills) == 2
    # full-height columns, in order
    assert fills[0].pts[:, 1].min() == -3.0 and fills[0].pts[:, 1].max() == 0.0
    assert fills[0].pts[:, 0].max() == 1.0
    assert fills[1].pts[:, 0].max() == 2.0
    # opacity is monotone in |v[j]|, and the peak column is fully opaque
    assert fills[0].opacity == pytest.approx(1.0)
    assert fills[1].opacity < fills[0].opacity
    assert fills[1].opacity >= 0.14


def test_rank1_rejects_factors_that_do_not_match_the_block():
    b = Block(3, 3)
    with pytest.raises(ValueError, match="v has 2 entries"):
        rank1(b, np.ones(3), np.ones(2))
    with pytest.raises(ValueError, match="u has 2 entries"):
        rank1(b, np.ones(2), np.ones(3))


def test_heat_covers_exactly_the_block_extent():
    V = np.arange(6, dtype=float).reshape(2, 3)
    b = Block(2, 3, values=V)
    r = heat(b)
    assert r.extent == b.extent
    assert np.array_equal(r.values, V)


def test_heat_needs_values():
    with pytest.raises(ValueError, match="carries no values"):
        heat(Block(2, 3))


def test_hinton_encodes_magnitude_as_area_not_as_side():
    # a 4x magnitude ratio must give a 4x AREA ratio, i.e. a 2x side ratio
    b = Block(1, 2, values=np.array([[1.0, 4.0]]))
    sq = hinton(b)
    assert len(sq) == 2
    side = [float(s.pts[:, 0].max() - s.pts[:, 0].min()) for s in sq]
    assert side[1] / side[0] == pytest.approx(2.0)


def test_hinton_splits_sign_across_two_roles_and_skips_zeros():
    b = Block(1, 3, values=np.array([[1.0, -1.0, 0.0]]))
    sq = hinton(b)
    assert len(sq) == 2                       # the zero emits nothing
    assert sq[0].role is Role.ACCENT1
    assert sq[1].role is Role.ACCENT2


def test_entries_labels_every_cell_at_its_center():
    b = Block(2, 2, values=np.array([[1.0, 2.0], [3.0, 4.0]]))
    labs = entries(b)
    assert len(labs) == 4
    assert all(isinstance(el, MathLabel) for el in labs)
    assert labs[0].anchor == b.center(0, 0)
    assert labs[0].latex == "1"


# --- the expression row --------------------------------------------------

def test_expr_places_blocks_left_to_right_on_a_common_center():
    A, B = Block(3, 2, name="A"), Block(2, 4, name="B")
    ops, placed = expr([A, "@", B, "=", Block(3, 4, name="C")],
                       cell=1.0, gap=0.5, op_gap=1.0)
    assert [p.name for p in placed] == ["A", "B", "C"]
    assert placed[0].extent[1] <= placed[1].extent[0]
    assert placed[1].extent[1] <= placed[2].extent[0]
    for p in placed:
        assert (p.extent[2] + p.extent[3]) / 2 == pytest.approx(0.0)


def test_expr_forces_one_shared_cell_so_inner_dimensions_line_up():
    A = Block(3, 2, cell=9.0, name="A")     # a stale cell must not survive
    B = Block(2, 4, cell=0.1, name="B")
    _, placed = expr([A, "@", B], cell=2.0)
    assert all(p.cell == 2.0 for p in placed)
    # A's 2 columns and B's 2 rows are drawn at the same physical size
    assert placed[0].width / placed[0].n == placed[1].height / placed[1].m


def test_expr_emits_one_label_per_operator():
    ops, placed = expr([Block(2, 2), "=", Block(2, 2), "+", Block(2, 2)])
    assert len(ops) == 2
    assert [o.latex for o in ops] == ["=", "+"]
    assert all(isinstance(o, MathLabel) for o in ops)
    assert placed[0].extent[1] <= ops[0].anchor[0] <= placed[1].extent[0]


def test_expr_preserves_values_through_placement():
    V = np.ones((2, 2))
    _, placed = expr([Block(2, 2, values=V, name="A")])
    assert np.array_equal(placed[0].values, V)


def test_bounds_covers_every_block_with_padding():
    _, placed = expr([Block(2, 2), "=", Block(2, 2)], cell=1.0)
    xlim, ylim = bounds(placed, pad=0.5)
    assert xlim[0] == pytest.approx(placed[0].extent[0] - 0.5)
    assert xlim[1] == pytest.approx(placed[-1].extent[1] + 0.5)
    assert ylim == pytest.approx((-1.5, 1.5))


# --- the gates -----------------------------------------------------------

def _fails(fn) -> list[str]:
    c = Checks()
    fn(c)
    return c.failures


def test_check_conformable_is_silent_on_a_valid_product():
    terms = [Block(3, 2, name="A"), "@", Block(2, 4, name="B"), "=",
             Block(3, 4, name="C")]
    assert _fails(lambda c: check_conformable(c, terms)) == []


def test_check_conformable_catches_a_mismatched_inner_dimension():
    terms = [Block(3, 2, name="A"), "@", Block(3, 4, name="B")]
    msgs = _fails(lambda c: check_conformable(c, terms))
    assert any("non-conformable" in m and "inner 2 != 3" in m for m in msgs)


def test_check_conformable_catches_summands_of_different_shape():
    terms = [Block(2, 2), "=", Block(2, 2), "+", Block(2, 3)]
    msgs = _fails(lambda c: check_conformable(c, terms))
    assert any("summands differ" in m for m in msgs)


def test_check_conformable_catches_unequal_sides_of_the_equals():
    terms = [Block(3, 4), "=", Block(3, 5)]
    msgs = _fails(lambda c: check_conformable(c, terms))
    assert any("sides differ" in m for m in msgs)


def test_check_cell_uniform_catches_a_block_placed_at_another_scale():
    _, placed = expr([Block(2, 2), "=", Block(2, 2)], cell=1.0)
    assert _fails(lambda c: check_cell_uniform(c, placed)) == []
    rogue = placed + [Block(2, 2, origin=(20.0, 0.0), cell=0.4, name="R")]
    msgs = _fails(lambda c: check_cell_uniform(c, rogue))
    assert any("cell" in m and "R" in m for m in msgs)


def test_check_no_overlap_catches_a_gap_too_small_for_the_operator():
    _, placed = expr([Block(2, 2), "=", Block(2, 2)], cell=1.0)
    assert _fails(lambda c: check_no_overlap(c, placed)) == []
    stacked = [Block(2, 2, name="A"), Block(2, 2, origin=(1.0, 0.0), name="B")]
    msgs = _fails(lambda c: check_no_overlap(c, stacked))
    assert any("overlap" in m for m in msgs)


def test_check_expr_proves_a_true_factorization():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((5, 3))
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    terms = [Block(5, 3, values=A, name="A"), "=",
             Block(5, 3, values=U, name="U"), "@",
             Block(3, 3, values=np.diag(s), name="S"), "@",
             Block(3, 3, values=Vt, name="Vt")]
    assert _fails(lambda c: check_expr(c, terms)) == []


def test_check_expr_catches_a_perturbed_factorization():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((5, 3))
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    s[0] *= 1.01                                  # a 1% lie
    terms = [Block(5, 3, values=A, name="A"), "=",
             Block(5, 3, values=U, name="U"), "@",
             Block(3, 3, values=np.diag(s), name="S"), "@",
             Block(3, 3, values=Vt, name="Vt")]
    msgs = _fails(lambda c: check_expr(c, terms))
    assert any("does not hold" in m for m in msgs)


def test_check_expr_sums_over_plus():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((4, 3))
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    terms = [Block(4, 3, values=A, name="A"), "="]
    for k in range(3):
        terms += [Block(4, 3, values=s[k] * np.outer(U[:, k], Vt[k, :]),
                        name=f"s{k}"), "+"]
    terms.pop()
    assert _fails(lambda c: check_expr(c, terms)) == []


def test_check_expr_refuses_blocks_without_values():
    terms = [Block(2, 2), "=", Block(2, 2)]
    msgs = _fails(lambda c: check_expr(c, terms))
    assert any("carries no values" in m for m in msgs)


def test_check_hinton_area_is_silent_on_a_correct_diagram():
    b = Block(2, 2, values=np.array([[1.0, -4.0], [9.0, 0.0]]))
    assert _fails(lambda c: check_hinton_area(c, b, hinton(b))) == []


def test_check_hinton_area_catches_side_proportional_to_magnitude():
    b = Block(1, 2, values=np.array([[1.0, 4.0]]))
    bad = []
    for j, v in ((0, 1.0), (1, 4.0)):
        cx, cy = b.center(0, j)
        h = 0.1 * v                          # side proportional to |v|
        bad.append(FilledCurve(np.array([[cx - h, cy - h], [cx + h, cy - h],
                                         [cx + h, cy + h], [cx - h, cy + h]])))
    msgs = _fails(lambda c: check_hinton_area(c, b, bad))
    assert any("area/|v| not constant" in m for m in msgs)
