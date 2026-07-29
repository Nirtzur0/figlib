"""Class B: the layout pass on top of schematic.py's nodes/ports/edges.

What is under test is the part that can be WRONG independently of taste:
edges must terminate on box boundaries (not centers), a rank layout must
produce boxes that do not overlap, an explicit position must beat the
computed one, and an auto-sized box must actually contain its own typeset
label. The benchmark figure's mechanical cleanliness is the integration
test — layout is the content of a Class B figure, so a collision there is
a content bug, not a cosmetic one.
"""

import numpy as np
import pytest

from figlib import schematic as sch
from figlib.style import Role


def _rects_overlap(a, b, pad=0.0):
    return (a[0] - pad < b[2] and b[0] - pad < a[2]
            and a[1] - pad < b[3] and b[1] - pad < a[3])


# --- boundary attachment ----------------------------------------------------


class TestBoundaryAttachment:
    def test_connect_lands_on_boundaries_not_centers(self):
        a = sch.Node("a", (0.0, 0.0), 1.0, 0.6)
        b = sch.Node("b", (3.0, 0.0), 1.0, 0.6)

        e = sch.connect(a, b, "map")

        assert e.anchors[0] == pytest.approx((0.5, 0.0))
        assert e.anchors[1] == pytest.approx((2.5, 0.0))
        assert a.boundary_distance(e.anchors[0]) < 1e-12
        assert b.boundary_distance(e.anchors[1]) < 1e-12
        # and emphatically NOT the centers
        assert np.hypot(*(np.subtract(e.anchors[0], a.center))) > 0.4

    def test_diagonal_connect_exits_through_the_nearer_face(self):
        a = sch.Node("a", (0.0, 0.0), 2.0, 0.4)   # wide and flat
        b = sch.Node("b", (0.0, 3.0), 2.0, 0.4)

        e = sch.connect(a, b, "map")

        # straight up: leaves through the top face, at the top y
        assert e.anchors[0] == pytest.approx((0.0, 0.2))
        assert e.anchors[1] == pytest.approx((0.0, 2.8))

    def test_curved_route_is_trimmed_back_to_the_boundary(self):
        a = sch.Node("a", (0.0, 0.0), 1.0, 0.6)
        b = sch.Node("b", (4.0, 0.0), 1.0, 0.6)

        e = sch.connect(a, b, "map", route="quad", via=(2.0, 1.5))

        assert a.boundary_distance(e.anchors[0]) < 1e-9
        assert b.boundary_distance(e.anchors[1]) < 1e-9
        # no drawn vertex sits strictly inside either box
        for nd in (a, b):
            interior = [p for p in e.pts if nd.boundary_distance(p) > 1e-9
                        and nd.contains(p)]
            assert not interior, f"{len(interior)} drawn points inside {nd.key}"

    def test_explicit_port_overrides_the_computed_face(self):
        a = sch.Node("a", (0.0, 0.0), 1.0, 0.6)
        b = sch.Node("b", (3.0, 0.0), 1.0, 0.6)

        e = sch.connect(a, b, "map", src_port="top", dst_port="bottom",
                        route="elbow", corner="vh")

        assert e.anchors[0] == pytest.approx(a.port("top"))
        assert e.anchors[1] == pytest.approx(b.port("bottom"))

    def test_boundary_toward_respects_pad(self):
        a = sch.Node("a", (0.0, 0.0), 1.0, 0.6)
        p = sch.boundary_toward(a, (5.0, 0.0), pad=0.1)
        assert p == pytest.approx((0.6, 0.0))

    def test_trim_drops_the_part_inside_the_box(self):
        a = sch.Node("a", (0.0, 0.0), 1.0, 0.6)
        # a polyline starting at the CENTER, walking out to the right
        pts = np.array([[0.0, 0.0], [0.25, 0.0], [2.0, 0.0]])

        out = sch.trim_at_boundary(pts, a, end="start")

        assert out[0] == pytest.approx((0.5, 0.0))
        assert out[-1] == pytest.approx((2.0, 0.0))
        assert len(out) == 2                      # the interior vertices are gone
        assert a.boundary_distance(out[0]) < 1e-12

    def test_trim_at_the_far_end_is_symmetric(self):
        a = sch.Node("a", (3.0, 0.0), 1.0, 0.6)
        pts = np.array([[0.0, 0.0], [2.9, 0.0], [3.0, 0.0]])

        out = sch.trim_at_boundary(pts, a, end="end")

        assert out[-1] == pytest.approx((2.5, 0.0))
        assert out[0] == pytest.approx((0.0, 0.0))

    def test_trim_is_a_no_op_when_already_outside(self):
        a = sch.Node("a", (0.0, 0.0), 1.0, 0.6)
        pts = np.array([[0.5, 0.0], [2.0, 0.0]])
        assert np.allclose(sch.trim_at_boundary(pts, a, end="start"), pts)


# --- ranking and layout -----------------------------------------------------


class TestRanking:
    def test_longest_path_ranks_a_chain(self):
        r = sch.longest_path_ranks(["a", "b", "c"],
                                   [("a", "b"), ("b", "c")])
        assert r == {"a": 0, "b": 1, "c": 2}

    def test_longest_path_not_shortest(self):
        # a -> d directly AND a -> b -> c -> d: d sits at the LONG rank,
        # otherwise the skip edge would have to run backwards
        r = sch.longest_path_ranks(
            ["a", "b", "c", "d"],
            [("a", "b"), ("b", "c"), ("c", "d"), ("a", "d")])
        assert r == {"a": 0, "b": 1, "c": 2, "d": 3}

    def test_cycle_is_refused(self):
        with pytest.raises(ValueError, match="cycle"):
            sch.longest_path_ranks(["a", "b"], [("a", "b"), ("b", "a")])


class TestRankLayout:
    DAG = [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")]
    KEYS = ["a", "b", "c", "d"]

    def _boxes(self, pos, w=1.4, h=0.5):
        return {k: (x - w / 2, y - h / 2, x + w / 2, y + h / 2)
                for k, (x, y) in pos.items()}

    def test_dag_boxes_are_distinct_and_disjoint(self):
        lay = sch.RankLayout(rank_gap=1.2, lane_gap=2.0)
        pos = sch.rank_positions(self.KEYS, layout=lay, dag=self.DAG)

        assert len(set(pos.values())) == len(self.KEYS)
        boxes = self._boxes(pos)
        for i, ki in enumerate(self.KEYS):
            for kj in self.KEYS[i + 1:]:
                assert not _rects_overlap(boxes[ki], boxes[kj]), f"{ki}/{kj}"

    def test_rank_sets_the_flow_axis_and_lanes_are_centered(self):
        lay = sch.RankLayout(rank_gap=1.2, lane_gap=2.0)
        pos = sch.rank_positions(self.KEYS, layout=lay, dag=self.DAG)

        # vertical flow: rank -> y, evenly spaced
        assert pos["a"][1] == pytest.approx(0.0)
        assert pos["b"][1] == pytest.approx(1.2)
        assert pos["c"][1] == pytest.approx(1.2)
        assert pos["d"][1] == pytest.approx(2.4)
        # a rank of one is centered; a rank of two straddles the axis evenly
        assert pos["a"][0] == pytest.approx(0.0)
        assert pos["d"][0] == pytest.approx(0.0)
        assert pos["b"][0] == pytest.approx(-1.0)
        assert pos["c"][0] == pytest.approx(+1.0)

    def test_horizontal_axis_swaps_the_roles(self):
        lay = sch.RankLayout(axis="horizontal", rank_gap=1.2, lane_gap=2.0)
        pos = sch.rank_positions(self.KEYS, layout=lay, dag=self.DAG)
        assert pos["d"][0] == pytest.approx(2.4)
        assert pos["d"][1] == pytest.approx(0.0)

    def test_explicit_lane_beats_even_spacing(self):
        lay = sch.RankLayout(rank_gap=1.0, lane_gap=1.0)
        pos = sch.rank_positions(self.KEYS, layout=lay, dag=self.DAG,
                                 lanes={"b": 0, "c": 3})
        assert pos["b"][0] == pytest.approx(0.0)
        assert pos["c"][0] == pytest.approx(3.0)

    def test_position_override_wins_over_rank_placement(self):
        lay = sch.RankLayout(rank_gap=1.2, lane_gap=2.0)
        base = sch.rank_positions(self.KEYS, layout=lay, dag=self.DAG)
        pos = sch.rank_positions(self.KEYS, layout=lay, dag=self.DAG,
                                 overrides={"d": (-7.5, 9.25)})

        assert pos["d"] == pytest.approx((-7.5, 9.25))
        assert base["d"] != pos["d"]
        # and nothing else moved: the override is local, not a re-solve
        for k in ("a", "b", "c"):
            assert pos[k] == pytest.approx(base[k])

    def test_explicit_ranks_are_used_verbatim(self):
        lay = sch.RankLayout(rank_gap=1.0, lane_gap=1.0)
        pos = sch.rank_positions(["a", "b"], layout=lay,
                                 ranks={"a": 5, "b": 2})
        assert pos["a"][1] == pytest.approx(5.0)
        assert pos["b"][1] == pytest.approx(2.0)


# --- auto-sizing from label metrics -----------------------------------------


class TestAutoSize:
    def test_auto_sized_box_contains_its_label_with_padding(self):
        scale = 100.0     # px per math unit
        latex = r"\text{Multi-Head Attention}"
        w, h = sch.auto_size(latex, scale, size_pt=11.0)
        lw, lh = sch.label_extent_px(latex, 11.0)

        assert w * scale > lw and h * scale > lh
        nd = sch.auto_node("mha", (0.0, 0.0), latex, scale=scale)
        assert (nd.width, nd.height) == pytest.approx((w, h))
        # the module's own honesty check agrees
        assert sch.label_overflow([nd], scale, 11.0) == []

    def test_min_size_gives_a_uniform_column(self):
        scale = 100.0
        labels = [r"\text{MLP}", r"\text{Multi-Head Attention}"]
        sizes = [sch.auto_size(t, scale) for t in labels]
        uniform = (max(w for w, _ in sizes), max(h for _, h in sizes))
        nodes = [sch.auto_node(t, (0.0, float(i)), t, scale=scale,
                               min_size=uniform)
                 for i, t in enumerate(labels)]

        assert nodes[0].width == pytest.approx(nodes[1].width)
        assert sch.label_overflow(nodes, scale, 11.0) == []

    def test_a_too_small_box_is_still_reported(self):
        # auto-sizing is opt-in; a hand-stated box that lies still fails
        nd = sch.Node("n", (0.0, 0.0), 0.2, 0.1, label=r"\text{LayerNorm}")
        assert sch.label_overflow([nd], 100.0, 11.0)

    def test_circle_node_is_round_and_holds_its_glyph(self):
        nd = sch.circle_node("add", (0.0, 0.0), 0.26, label="+")
        assert nd.width == pytest.approx(nd.height) == pytest.approx(0.52)
        assert nd.radius == pytest.approx(0.26)
        assert sch.label_overflow([nd], 100.0, 11.0) == []


# --- assembly ---------------------------------------------------------------


class TestAssemble:
    def test_boxes_paint_over_edges_and_edge_labels_over_boxes(self):
        from figlib.scene import Curve, FilledCurve, MathLabel

        a = sch.Node("a", (0.0, 0.0), 1.0, 0.6, label="a", fill="#ffffff")
        b = sch.Node("b", (3.0, 0.0), 1.0, 0.6, label="b", fill="#ffffff")
        e = sch.connect(a, b, "map", label=r"d")

        out = sch.assemble([a, b], [e])
        kinds = [type(it).__name__ for it in out]

        i_edge = kinds.index("Curve")
        i_box = min(i for i, it in enumerate(out)
                    if isinstance(it, FilledCurve))
        i_edge_label = max(i for i, it in enumerate(out)
                           if isinstance(it, MathLabel) and it.latex == "d")
        assert i_edge < i_box < i_edge_label

    def test_under_and_over_bracket_the_body(self):
        from figlib.scene import MathLabel

        a = sch.Node("a", (0.0, 0.0), 1.0, 0.6)
        rail = sch.rails([0.0], (-1.0, 1.0))
        top = MathLabel("t", (0.0, 2.0))
        out = sch.assemble([a], [], under=rail, over=[top])
        assert out[0] is rail[0]
        assert out[-1] is top


# --- regions: grouping as a filled ground -----------------------------------
#
# The transformer-circuits idiom: containment is a pale filled rounded rect
# BEHIND the nodes, and nesting is read off the contrast ladder rather than
# off any drawn line. What can be wrong independently of taste is the
# structure — a region must contain what it claims to contain, two regions
# must nest or be disjoint (a half-overlap groups nothing), and the paint
# order must put a child over its parent and both under the nodes.


class TestRegion:
    def test_enclose_derives_the_bbox_of_its_members_with_pad(self):
        a = sch.Node("a", (0.0, 0.0), 1.0, 0.6)
        b = sch.Node("b", (3.0, 1.0), 1.0, 0.6)

        r = sch.enclose("g", [a, b], pad=0.2)

        assert r.rect == pytest.approx((-0.7, -0.5, 3.7, 1.5))
        assert r.members == ("a", "b")

    def test_enclose_accepts_points_and_nested_regions(self):
        a = sch.Node("a", (0.0, 0.0), 1.0, 0.6)
        inner = sch.enclose("inner", [a], pad=0.1)

        outer = sch.enclose("outer", [inner, (2.0, 2.0)], pad=0.0)

        assert outer.rect == pytest.approx((-0.6, -0.4, 2.0, 2.0))
        # a point contributes no key; only named members are checkable
        assert outer.members == ("inner",)

    def test_a_region_is_a_fill_that_edges_may_cross(self):
        from figlib.scene import FilledCurve

        a = sch.Node("a", (0.0, 0.0), 1.0, 0.6)
        r = sch.enclose("g", [a], pad=0.2)

        (fill,) = [it for it in r.items() if isinstance(it, FilledCurve)]
        assert fill.outline is False
        assert fill.role is Role.CONSTRUCTION
        # NOT a Node: clearance_violations must never see it, so an edge
        # running through a group is not a collision
        assert not isinstance(r, sch.Node)

    def test_default_opacity_clears_the_colour_gate_on_both_house_themes(self):
        """The circuits corpus washes its groups at ~1.03:1, which the house
        floor forbids on textured paper. The default is pinned to the floor
        instead, and this is the test that keeps it there."""
        from figlib.color import composite, contrast
        from figlib.gates import MIN_PERCEPTIBLE_CONTRAST
        from figlib.style import DEFAULT_STYLE
        from figlib.theme import RISO

        for th in (RISO, DEFAULT_STYLE):
            papers = (th.paper_stops() if hasattr(th, "paper_stops")
                      else [th.background])
            ink = th.ink(Role.CONSTRUCTION).color
            worst = min(contrast(composite(ink, sch.REGION_OPACITY, p), p)
                        for p in papers)
            assert worst >= MIN_PERCEPTIBLE_CONTRAST

    def test_containment_violation_names_the_member_that_escaped(self):
        a = sch.Node("a", (0.0, 0.0), 1.0, 0.6)
        b = sch.Node("b", (3.0, 0.0), 1.0, 0.6)
        r = sch.enclose("g", [a], pad=0.2)
        r = sch.Region(r.key, r.center, r.width, r.height, members=("a", "b"))

        bad = sch.region_containment_violations([r], [a, b])

        assert len(bad) == 1 and "'b'" in bad[0] and "'g'" in bad[0]

    def test_a_fully_contained_member_is_clean(self):
        a = sch.Node("a", (0.0, 0.0), 1.0, 0.6)
        r = sch.enclose("g", [a], pad=0.2)

        assert sch.region_containment_violations([r], [a]) == []

    def test_partially_overlapping_regions_are_a_violation(self):
        a = sch.Region("a", (0.0, 0.0), 2.0, 2.0)
        b = sch.Region("b", (1.0, 1.0), 2.0, 2.0)

        bad = sch.region_nesting_violations([a, b])

        assert len(bad) == 1 and "'a'" in bad[0] and "'b'" in bad[0]

    def test_nested_and_disjoint_regions_are_both_clean(self):
        outer = sch.Region("outer", (0.0, 0.0), 4.0, 4.0)
        inner = sch.Region("inner", (0.5, 0.5), 1.0, 1.0)
        far = sch.Region("far", (10.0, 0.0), 2.0, 2.0)

        assert sch.region_nesting_violations([outer, inner, far]) == []

    def test_assemble_paints_regions_under_edges_child_over_parent(self):
        from figlib.scene import Curve, FilledCurve

        a = sch.Node("a", (0.0, 0.0), 1.0, 0.6, fill="#ffffff")
        b = sch.Node("b", (3.0, 0.0), 1.0, 0.6, fill="#ffffff")
        e = sch.connect(a, b, "map")
        inner = sch.enclose("inner", [a], pad=0.2)
        outer = sch.enclose("outer", [a, b], pad=0.5)

        # deliberately passed smallest-first: draw order is derived, not given
        out = sch.assemble([a, b], [e], regions=[inner, outer])

        fills = [i for i, it in enumerate(out) if isinstance(it, FilledCurve)]
        i_outer, i_inner = fills[0], fills[1]
        i_edge = next(i for i, it in enumerate(out) if isinstance(it, Curve))
        i_box = fills[2]
        assert i_outer < i_inner < i_edge < i_box

    def test_a_region_label_sits_inside_its_own_top_edge(self):
        from figlib.scene import MathLabel

        a = sch.Node("a", (0.0, 0.0), 1.0, 0.6)
        r = sch.enclose("g", [a], pad=0.3, label=r"\text{group}")

        (lab,) = [it for it in r.items() if isinstance(it, MathLabel)]
        x0, y0, x1, y1 = r.rect
        assert lab.anchor == pytest.approx((0.0, y1))
        assert lab.va == "top" and lab.role is Role.ANNOTATION


# --- the benchmark figure ---------------------------------------------------


class TestTransformerBlockBenchmark:
    @staticmethod
    def _load():
        from pathlib import Path

        from figlib.program import load_program
        root = Path(__file__).resolve().parents[1]
        return load_program(root / "figures" / "circuits" / "schematic_transformer_block.py")

    def test_build_passes_the_mechanical_gate_with_zero_diagnostics(self):
        from figlib.gates import mechanical

        mod = self._load()
        scene = mod.build(mod.compute(mod.PARAMS))
        style = mod.THEME.scaled(mod.FORMAT.ink_scale)

        diags = mechanical(scene, style, width_px=mod.FORMAT.display_width_px)
        assert diags == [], "\n".join(f"{d.kind}: {d.detail}" for d in diags)

    def test_numerical_assertions_hold(self):
        mod = self._load()
        mod.assertions(mod.compute(mod.PARAMS))

    def test_the_spine_is_the_heaviest_ink_and_vertical(self):
        mod = self._load()
        g = mod.compute(mod.PARAMS)
        spine = [e for e in g["edges"] if e.key.startswith("spine")]
        branch = [e for e in g["edges"] if not e.key.startswith("spine")]

        assert spine and branch
        assert min(e.width_scale for e in spine) > max(e.width_scale for e in branch)
        for e in spine:
            assert np.allclose(e.pts[:, 0], e.pts[0, 0]), f"{e.key} is not vertical"
            assert e.role is Role.CONTENT

    def test_renders_under_riso(self, tmp_path):
        from figlib.render import save
        from figlib.theme import RISO

        mod = self._load()
        scene = mod.build(mod.compute(mod.PARAMS))
        svg, png = save(scene, tmp_path / "block", RISO,
                        width_px=mod.FORMAT.display_width_px)
        assert svg.exists() and png.stat().st_size > 0


# ===========================================================================
# The core layer: ports, routing, edge-kind decoration, and the schematic
# checks the induction-head benchmark asserts against.
# ===========================================================================

from figlib.scene import Curve, FilledCurve, MathLabel     # noqa: E402


def _node(**kw):
    kw.setdefault("key", "n")
    kw.setdefault("center", (0.0, 0.0))
    kw.setdefault("width", 2.0)
    kw.setdefault("height", 1.0)
    return sch.Node(**kw)


class TestRoundedRect:
    def test_square_corners_are_the_four_vertices(self):
        pts = sch.rounded_rect((1.0, 2.0), 4.0, 2.0, radius=0.0)
        assert sorted(map(tuple, pts)) == [(-1.0, 1.0), (-1.0, 3.0),
                                           (3.0, 1.0), (3.0, 3.0)]

    def test_rounded_outline_stays_inside_and_touches_every_side(self):
        pts = sch.rounded_rect((0.0, 0.0), 4.0, 2.0, radius=0.5, n_arc=16)
        assert np.max(np.abs(pts[:, 0])) == pytest.approx(2.0)
        assert np.max(np.abs(pts[:, 1])) == pytest.approx(1.0)
        assert np.all(np.abs(pts[:, 0]) <= 2.0 + 1e-12)
        assert np.all(np.abs(pts[:, 1]) <= 1.0 + 1e-12)

    def test_radius_clamps_to_half_the_short_side_giving_a_pill(self):
        pts = sch.rounded_rect((0.0, 0.0), 4.0, 2.0, radius=99.0, n_arc=32)
        # every point is exactly 1.0 from the spine segment y = 0, |x| <= 1
        r = np.hypot(pts[:, 0] - np.clip(pts[:, 0], -1.0, 1.0), pts[:, 1])
        assert np.allclose(r, 1.0)


class TestPorts:
    def test_the_four_side_midpoints(self):
        n = _node(center=(1.0, 2.0), width=2.0, height=1.0)
        assert n.port("left") == pytest.approx((0.0, 2.0))
        assert n.port("right") == pytest.approx((2.0, 2.0))
        assert n.port("bottom") == pytest.approx((1.0, 1.5))
        assert n.port("top") == pytest.approx((1.0, 2.5))

    def test_fractional_position_runs_low_to_high_on_every_side(self):
        n = _node(center=(0.0, 0.0), width=4.0, height=2.0)
        assert n.port("left@0.25") == pytest.approx((-2.0, -0.5))
        assert n.port("right@0.75") == pytest.approx((2.0, 0.5))
        assert n.port("bottom@0.25") == pytest.approx((-1.0, -1.0))
        assert n.port("top@0.75") == pytest.approx((1.0, 1.0))

    def test_tuple_spec_matches_string_spec(self):
        assert _node().port(("left", 0.2)) == pytest.approx(_node().port("left@0.2"))

    def test_endpoints_of_a_side_are_its_corners(self):
        n = _node(center=(0.0, 0.0), width=4.0, height=2.0)
        assert n.port("top@0") == pytest.approx((-2.0, 1.0))
        assert n.port("top@1") == pytest.approx((2.0, 1.0))

    def test_unknown_side_is_rejected(self):
        with pytest.raises(ValueError):
            _node().port("north")

    def test_every_port_sits_on_the_boundary(self):
        n = _node(center=(3.0, -1.0), width=1.7, height=0.9, corner=0.3)
        for side in ("left", "right", "top", "bottom"):
            for t in (0.0, 0.3, 0.5, 1.0):
                assert n.boundary_distance(n.port((side, t))) < 1e-12

    def test_boundary_distance_is_unsigned_inside_and_out(self):
        n = _node(center=(0.0, 0.0), width=4.0, height=2.0)
        assert n.boundary_distance((0.0, 0.0)) == pytest.approx(1.0)
        assert n.boundary_distance((3.0, 0.0)) == pytest.approx(1.0)
        assert n.boundary_distance((3.0, 2.0)) == pytest.approx(np.hypot(1.0, 1.0))
        assert n.contains((1.9, 0.9)) and not n.contains((2.1, 0.0))
        assert n.contains((2.1, 0.0), pad=0.2)


class TestNodeItems:
    def test_open_node_is_a_closed_curve_plus_its_centred_label(self):
        n = _node(label="x", role=Role.ACCENT1)
        outline, label = n.items()
        assert isinstance(outline, Curve) and outline.closed
        assert outline.role is Role.ACCENT1
        assert isinstance(label, MathLabel)
        assert label.anchor == n.center and label.ha == "center" and label.va == "center"

    def test_filled_node_is_a_paper_filled_outlined_region(self):
        body = _node(label="x", fill="#eeeeee").items()[0]
        assert isinstance(body, FilledCurve)
        assert body.color == "#eeeeee" and body.outline and body.opacity == 1.0

    def test_unlabelled_node_emits_only_its_body(self):
        assert len(_node().items()) == 1


class TestRouting:
    def test_straight_is_two_points(self):
        assert sch.straight((0.0, 0.0), (3.0, 4.0)).shape == (2, 2)

    def test_elbow_bends_once_and_stays_axis_aligned(self):
        hv = sch.elbow((0.0, 0.0), (3.0, 4.0), "hv")
        assert hv[1] == pytest.approx((3.0, 0.0))
        vh = sch.elbow((0.0, 0.0), (3.0, 4.0), "vh")
        assert vh[1] == pytest.approx((0.0, 4.0))
        for pts in (hv, vh):
            for a, b in zip(pts[:-1], pts[1:]):
                assert abs(a[0] - b[0]) < 1e-12 or abs(a[1] - b[1]) < 1e-12

    def test_elbow_takes_an_explicit_corner(self):
        assert sch.elbow((0.0, 0.0), (3.0, 4.0), (1.0, 1.0))[1] == pytest.approx((1.0, 1.0))

    def test_quad_passes_through_its_via_point(self):
        p, q, via = (0.0, 0.0), (4.0, 0.0), (2.0, 3.0)
        pts = sch.quad_through(p, q, via, n=101)
        assert pts[0] == pytest.approx(p)
        assert pts[-1] == pytest.approx(q)
        assert pts[50] == pytest.approx(via)            # t = 1/2, by construction

    def test_cubic_passes_through_both_via_points(self):
        pts = sch.cubic_through((0.0, 0.0), (6.0, 0.0), (2.0, 2.0), (4.0, -2.0), n=97)
        assert pts[32] == pytest.approx((2.0, 2.0))     # t = 1/3
        assert pts[64] == pytest.approx((4.0, -2.0))    # t = 2/3

    def test_route_pts_dispatch_and_its_errors(self):
        assert len(sch.route_pts((0, 0), (1, 1), "elbow")) == 3
        with pytest.raises(ValueError):
            sch.route_pts((0, 0), (1, 1), "quad")
        with pytest.raises(ValueError):
            sch.route_pts((0, 0), (1, 1), "spline")

    def test_edge_records_the_ports_it_was_given(self):
        e = sch.edge((0.0, 0.0), (4.0, 0.0), "excite", route="quad", via=(2.0, 1.0))
        assert e.anchors[0] == pytest.approx((0.0, 0.0))
        assert e.anchors[1] == pytest.approx((4.0, 0.0))

    def test_unknown_kind_is_rejected_at_construction(self):
        with pytest.raises(ValueError):
            sch.edge((0, 0), (1, 0), "inhibits")


class TestDecorationPolicy:
    """Each kind's visual identity, read back off the emitted Curve."""

    def _e(self, kind, **kw):
        return sch.edge((0.0, 0.0), (4.0, 0.0), kind, **kw)

    def test_excite_is_a_solid_stroke_with_one_filled_head(self):
        [c] = self._e("excite").items()
        assert c.dash == "solid" and c.arrow_style == "filled" and c.arrows == (1.0,)

    def test_inhibit_has_no_arrowhead_but_a_flat_terminal_bar(self):
        stroke, bar = self._e("inhibit", bar_half=0.25).items()
        assert stroke.arrows == ()
        # perpendicular to the incoming direction, centred on the tip
        assert bar.pts[:, 0] == pytest.approx([4.0, 4.0])
        assert sorted(bar.pts[:, 1]) == pytest.approx([-0.25, 0.25])

    def test_map_is_a_solid_stroke_with_a_hollow_head(self):
        [c] = self._e("map").items()
        assert c.arrow_style == "hollow" and c.arrows == (1.0,) and c.dash == "solid"

    def test_attend_is_dashed_hollow_and_thinner(self):
        [c] = self._e("attend").items()
        assert c.dash == "dashed" and c.arrow_style == "hollow"
        assert c.width_scale == pytest.approx(0.85)

    def test_copy_stacks_two_filled_chevrons(self):
        [c] = self._e("copy", chevron_gap=0.4).items()
        assert c.arrow_style == "filled" and len(c.arrows) == 2
        assert c.arrows[1] == pytest.approx(1.0)
        assert c.arrows[0] == pytest.approx(1.0 - 0.4 / 4.0)

    def test_chevron_spacing_is_a_fixed_distance_not_a_fixed_fraction(self):
        """A fractional gap would drift to mid-curve on a long edge and read
        as a waypoint rather than a doubled head."""
        gap = 0.5
        for length in (2.0, 20.0):
            fr = sch.edge((0.0, 0.0), (length, 0.0), "copy",
                          chevron_gap=gap).head_fractions()
            assert (1.0 - fr[0]) * length == pytest.approx(gap)

    def test_dash_override_beats_the_kind_policy(self):
        assert self._e("copy", dash="dotted").items()[0].dash == "dotted"

    def test_role_and_colour_pass_through_untouched(self):
        [c] = self._e("excite", role=Role.ACCENT2, color="#123456",
                      opacity=0.4).items()
        assert c.role is Role.ACCENT2 and c.color == "#123456"
        assert c.opacity == pytest.approx(0.4)

    def test_edge_label_rides_the_curve_or_an_explicit_anchor(self):
        e = sch.edge((0.0, 0.0), (4.0, 0.0), "excite", label="f", label_at=0.25)
        assert e.items()[-1].anchor == pytest.approx((1.0, 0.0))
        lab = sch.edge((0.0, 0.0), (4.0, 0.0), "excite", label="f",
                       label_anchor=(9.0, 9.0), label_halo=True).items()[-1]
        assert lab.anchor == (9.0, 9.0) and lab.halo

    def test_every_kind_emits_curves_the_renderer_understands(self):
        for kind in sch.EDGE_KINDS:
            for it in self._e(kind).items():
                assert isinstance(it, (Curve, MathLabel))
                if isinstance(it, Curve):
                    assert it.arrow_style in ("filled", "hollow")


class TestRailsAndPlacement:
    def test_vertical_rails_span_the_extent_as_construction_ink(self):
        items = sch.rails([0.0, 1.0, 2.0], (-1.0, 3.0))
        assert len(items) == 3
        for c, x in zip(items, [0.0, 1.0, 2.0]):
            assert isinstance(c, Curve) and c.role is Role.CONSTRUCTION
            assert np.allclose(c.pts[:, 0], x)
            assert c.pts[0, 1] == -1.0 and c.pts[1, 1] == 3.0

    def test_horizontal_rails_swap_the_axes(self):
        [c] = sch.rails([2.0], (0.0, 5.0), axis="horizontal")
        assert np.allclose(c.pts[:, 1], 2.0)
        assert c.pts[0, 0] == 0.0 and c.pts[1, 0] == 5.0

    def test_headers_land_at_the_low_end_by_default(self):
        items = sch.rails([0.0, 1.0], (-1.0, 3.0), labels=["a", "b"])
        labels = items[2:]
        assert len(items[:2]) == 2 and len(labels) == 2
        assert all(isinstance(m, MathLabel) and m.va == "top" for m in labels)
        assert [m.anchor for m in labels] == [(0.0, -1.0), (1.0, -1.0)]

    def test_headers_can_go_to_the_high_end(self):
        lab = sch.rails([0.0], (-1.0, 3.0), labels=["a"], label_end="high")[1]
        assert lab.anchor == (0.0, 3.0) and lab.va == "bottom"

    def test_bad_axis_is_rejected(self):
        with pytest.raises(ValueError):
            sch.rails([0.0], (0.0, 1.0), axis="diagonal")

    def test_columns_and_rows_are_plain_affine(self):
        assert sch.columns(4, 1.0, 2.5) == [1.0, 3.5, 6.0, 8.5]
        assert sch.rows(3, -1.0, 0.5) == [-1.0, -0.5, 0.0]

    def test_grid_maps_indices_to_centres(self):
        g = sch.Grid(origin=(1.0, 2.0), dx=3.0, dy=-0.5)
        assert g.center(0, 0) == (1.0, 2.0)
        assert g.center(2, 3) == (7.0, 0.5)

    def test_items_flattens_nodes_edges_and_raw_items(self):
        n = _node(label="x")
        e = sch.edge(n.port("right"), (5.0, 0.0), "excite")
        extra = MathLabel("z", (0.0, 0.0))
        out = sch.items([n], e, extra)
        assert len(out) == 2 + 1 + 1 and out[-1] is extra


class TestSchematicChecks:
    def test_clearance_fires_when_an_edge_pierces_an_unrelated_box(self):
        blocker = _node(key="blocker", center=(2.0, 0.0), width=1.0, height=1.0)
        e = sch.edge((0.0, 0.0), (4.0, 0.0), "excite", key="through", n=64,
                     route="quad", via=(2.0, 0.0))
        bad = sch.clearance_violations([e], [blocker])
        assert len(bad) == 1 and "blocker" in bad[0] and "through" in bad[0]

    def test_no_violation_for_the_boxes_the_edge_attaches_to(self):
        a = _node(key="a", center=(0.0, 0.0), width=1.0, height=1.0)
        b = _node(key="b", center=(5.0, 0.0), width=1.0, height=1.0)
        e = sch.edge(a.port("right"), b.port("left"), "excite", key="a->b")
        assert sch.clearance_violations([e], [a, b]) == []

    def test_pad_makes_the_clearance_check_stricter(self):
        near = _node(key="near", center=(2.0, 0.6), width=1.0, height=1.0)
        e = sch.edge((0.0, 0.0), (4.0, 0.0), "excite", key="e", n=64)
        assert sch.clearance_violations([e], [near]) == []
        assert len(sch.clearance_violations([e], [near], pad=0.2)) == 1

    def test_port_offsets_measure_distance_to_the_nearest_boundary(self):
        a = _node(key="a", center=(0.0, 0.0), width=2.0, height=1.0)
        b = _node(key="b", center=(6.0, 0.0), width=2.0, height=1.0)
        exact = sch.edge(a.port("right"), b.port("left"), "excite", key="exact")
        loose = sch.edge(a.port("right"), (4.5, 0.0), "excite", key="loose")
        assert sch.max_port_offset([exact], [a, b]) < 1e-12
        assert sch.max_port_offset([loose], [a, b]) == pytest.approx(0.5)

    def test_attached_nodes_reports_both_ends(self):
        a = _node(key="a", center=(0.0, 0.0), width=2.0, height=1.0)
        b = _node(key="b", center=(6.0, 0.0), width=2.0, height=1.0)
        e = sch.edge(a.port("right"), b.port("left"), "excite")
        assert sorted(sch.attached_nodes(e, [a, b], 1e-9)) == ["a", "b"]

    def test_crossing_count_on_a_known_configuration(self):
        x1 = sch.edge((0.0, 0.0), (2.0, 2.0), "excite", key="up")
        x2 = sch.edge((0.0, 2.0), (2.0, 0.0), "excite", key="down")
        par = sch.edge((0.0, 3.0), (2.0, 3.0), "excite", key="par")
        assert sch.crossing_count([x1, x2]) == 1
        (a, b, pt), = sch.crossings([x1, x2])
        assert {a, b} == {"up", "down"} and pt == pytest.approx((1.0, 1.0))
        assert sch.crossing_count([x1, par]) == 0
        assert sch.crossing_count([x1, x2, par]) == 1

    def test_edges_sharing_a_port_are_joined_not_crossed(self):
        a = sch.edge((0.0, 0.0), (2.0, 2.0), "excite", key="a")
        b = sch.edge((0.0, 0.0), (2.0, -2.0), "excite", key="b")
        assert sch.crossing_count([a, b]) == 0

    def test_a_curve_crossing_a_line_twice_counts_twice(self):
        arc = sch.edge((0.0, 0.0), (4.0, 0.0), "excite", route="quad",
                       via=(2.0, 2.0), n=201, key="arc")
        line = sch.edge((-1.0, 1.0), (5.0, 1.0), "excite", key="line")
        assert sch.crossing_count([arc, line]) == 2

    def test_label_overflow_flags_a_label_wider_than_its_declared_box(self):
        latex = r"\text{a very long node label indeed}"
        w, h = sch.label_extent_px(latex, 11.0)
        tight = _node(key="tight", width=w / 200.0, height=2 * h / 100.0, label=latex)
        roomy = _node(key="roomy", width=2 * w / 100.0, height=2 * h / 100.0, label=latex)
        bad = sch.label_overflow([tight, roomy], scale=100.0, size_pt=11.0)
        assert len(bad) == 1 and bad[0].startswith("tight")

    def test_px_per_unit_matches_the_real_transform(self):
        from figlib.layout import Transform
        from figlib.scene import Scene
        t = Transform(Scene(xlim=(-1.0, 4.0), ylim=(0.0, 2.0)), width_px=600)
        assert sch.px_per_unit((-1.0, 4.0), 600) == pytest.approx(t.scale_x)


class TestPointAt:
    def test_arc_length_fraction_on_an_L_shaped_polyline(self):
        pts = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 1.0]])
        assert sch.point_at(pts, 0.0) == pytest.approx((0.0, 0.0))
        assert sch.point_at(pts, 0.5) == pytest.approx((2.0, 0.0))
        assert sch.point_at(pts, 1.0) == pytest.approx((3.0, 1.0))

    def test_degenerate_polyline_returns_its_only_point(self):
        assert sch.point_at(np.array([[1.0, 2.0], [1.0, 2.0]]), 0.7) == pytest.approx((1.0, 2.0))


class TestInductionHeadBenchmark:
    @staticmethod
    def _load():
        from pathlib import Path

        from figlib.program import load_program
        root = Path(__file__).resolve().parents[1]
        return load_program(root / "figures" / "circuits" / "induction_head_circuit.py")

    def test_numerical_assertions_hold(self):
        mod = self._load()
        mod.assertions(mod.compute(mod.PARAMS))

    def test_mechanical_and_colour_gates_are_clean(self):
        from figlib.gates import color_gate, mechanical

        mod = self._load()
        scene = mod.build(mod.compute(mod.PARAMS))
        style = mod.THEME.scaled(mod.FORMAT.ink_scale)
        diags = (mechanical(scene, style, width_px=mod.FORMAT.display_width_px)
                 + color_gate(scene, style))
        assert diags == [], "\n".join(f"{d.kind}: {d.detail}" for d in diags)

    def test_the_two_heads_are_separable_by_hue_and_by_decoration(self):
        mod = self._load()
        g = mod.compute(mod.PARAMS)
        by_key = {e.key: e for e in g["edges"]}
        prev = by_key[f"prev{g['earlier']}->{g['key_pos']}"]
        assert prev.role is Role.ACCENT1
        assert by_key["QK"].role is by_key["OV"].role is Role.ACCENT2
        assert by_key["QK"].kind == "attend" and by_key["OV"].kind == "copy"
        # the lookup is dashed and hollow-headed; the copies are solid chevrons
        assert by_key["QK"].items()[0].dash == "dashed"
        assert by_key["OV"].items()[0].dash == "solid"
        assert len(by_key["OV"].items()[0].arrows) == 2

    def test_the_ensemble_is_muted_and_exactly_one_write_is_accented(self):
        mod = self._load()
        g = mod.compute(mod.PARAMS)
        writes = [e for e in g["edges"] if e.key.startswith("prev")]
        assert len(writes) == len(g["tokens"]) - 1
        assert sum(e.role is Role.ACCENT1 for e in writes) == 1
        assert all(e.role is Role.MUTED for e in writes if e.role is not Role.ACCENT1)


# ===========================================================================
# Honesty marks: the three elisions a schematic is allowed to make, drawn
# instead of confessed in a docstring — a stack of ghost cards behind a node
# that abbreviates many, a dashed `???` box for a mechanism nobody knows,
# and a diamond-plus-ellipsis terminator for an edge that is being cut.
# ===========================================================================


class TestElisionStack:
    def test_stack_emits_ghost_cards_behind_the_body(self):
        n = sch.Node("mha", (0.0, 0.0), 4.0, 2.0, label="MHA", stack=2)
        curves = [it for it in n.items() if isinstance(it, (Curve, FilledCurve))]

        # 2 ghosts + 1 body outline, ghosts first (drawn behind)
        assert len(curves) == 3
        assert curves[0].role is Role.MUTED and curves[1].role is Role.MUTED
        assert curves[2].role is Role.CONTENT

        off = sch.STACK_OFFSET * min(4.0, 2.0)
        assert np.allclose(curves[1].pts, curves[2].pts + [off, -off])
        assert np.allclose(curves[0].pts, curves[2].pts + [2 * off, -2 * off])

    def test_stack_offset_is_a_tenth_of_the_short_side(self):
        assert sch.STACK_OFFSET == pytest.approx(0.10)

    def test_no_stack_is_the_untouched_single_box(self):
        assert len(sch.Node("n", (0.0, 0.0), 2.0, 1.0).items()) == 1

    def test_ghosts_of_a_filled_node_are_paper_filled_too(self):
        n = sch.Node("n", (0.0, 0.0), 2.0, 1.0, fill="#eeeeee", stack=1)
        bodies = [it for it in n.items() if isinstance(it, (Curve, FilledCurve))]

        assert len(bodies) == 2
        ghost, body = bodies
        assert isinstance(ghost, FilledCurve) and ghost.color == "#eeeeee"
        assert ghost.role is Role.MUTED and ghost.outline
        assert isinstance(body, FilledCurve)

    def test_the_label_is_drawn_once_over_the_whole_stack(self):
        labels = [it for it in sch.Node("n", (0.0, 0.0), 2.0, 1.0, label="x",
                                        stack=3).items()
                  if isinstance(it, MathLabel)]
        assert len(labels) == 1


class TestUnknownMechanismNode:
    def test_unknown_node_is_dashed_with_a_mystery_label(self):
        n = sch.unknown_node("mystery", (0.0, 0.0), 3.0, 1.5)
        its = n.items()

        labels = [it for it in its if isinstance(it, MathLabel)]
        assert labels and "?" in labels[0].latex
        curves = [it for it in its if isinstance(it, Curve)]
        assert any(c.dash == "dashed" for c in curves)

    def test_unknown_node_passes_the_rest_through(self):
        n = sch.unknown_node("m", (1.0, 2.0), 3.0, 1.5, role=Role.MUTED, stack=1)
        assert n.center == (1.0, 2.0) and n.role is Role.MUTED and n.stack == 1

    def test_dash_on_an_open_node_lands_on_its_outline(self):
        (outline,) = sch.Node("n", (0.0, 0.0), 2.0, 1.0, dash="dashed").items()
        assert isinstance(outline, Curve) and outline.dash == "dashed"

    def test_a_filled_dashed_node_splits_into_fill_plus_dashed_outline(self):
        # FilledCurve has no dash channel, so the dash has to be its own
        # stroke over an outline-free fill.
        its = sch.Node("n", (0.0, 0.0), 2.0, 1.0, fill="#eeeeee",
                       dash="dashed").items()
        fill, outline = its[0], its[1]
        assert isinstance(fill, FilledCurve) and not fill.outline
        assert isinstance(outline, Curve) and outline.closed
        assert outline.dash == "dashed"
        assert np.allclose(outline.pts, fill.pts)


class TestDeclaredTruncation:
    def test_truncated_edge_has_no_head_but_a_diamond_and_an_ellipsis(self):
        e = sch.edge((0.0, 0.0), (10.0, 0.0), "map", truncated=True)
        its = e.items()

        main = its[0]
        assert main.arrows == ()          # no head — the diamond replaces it
        assert e.head_fractions() == ()

        labels = [it for it in its if isinstance(it, MathLabel)]
        assert any(r"\cdots" in lb.latex for lb in labels)

        diamonds = [it for it in its if isinstance(it, Curve) and it.closed
                    and len(it.pts) == 4]
        assert len(diamonds) == 1
        assert np.allclose(np.mean(diamonds[0].pts, axis=0), [10.0, 0.0],
                           atol=1e-9)

    def test_the_diamond_is_hollow_and_axis_aligned_to_the_terminal_tangent(self):
        e = sch.edge((0.0, 0.0), (0.0, 4.0), "map", truncated=True)
        (dia,) = [it for it in e.items() if isinstance(it, Curve) and it.closed]

        h = sch.TRUNC_DIAMOND_HALF
        # hollow: a stroked closed Curve, not a FilledCurve
        assert not any(isinstance(it, FilledCurve) for it in e.items())
        # tangent is +y here, so the vertices are the tip +/- h along x and y
        assert sorted(map(tuple, np.round(dia.pts, 12))) == [
            (-h, 4.0), (0.0, 4.0 - h), (0.0, 4.0 + h), (h, 4.0)]

    def test_the_ellipsis_sits_past_the_tip_along_the_tangent(self):
        e = sch.edge((0.0, 0.0), (10.0, 0.0), "map", truncated=True)
        (lb,) = [it for it in e.items() if isinstance(it, MathLabel)]
        assert lb.role is Role.ANNOTATION
        assert lb.anchor == pytest.approx((10.0 + 2.5 * sch.TRUNC_DIAMOND_HALF, 0.0))

    def test_diamond_half_diagonal_is_in_math_units_like_the_bar(self):
        # BAR_HALF's units: math, not px. A diamond an order of magnitude
        # bigger than the inhibit bar would be a different mark entirely.
        assert 0.2 * sch.BAR_HALF < sch.TRUNC_DIAMOND_HALF < 5.0 * sch.BAR_HALF

    def test_a_truncated_inhibit_drops_its_terminal_bar_too(self):
        e = sch.edge((0.0, 0.0), (3.0, 0.0), "inhibit", truncated=True)
        closed = [it for it in e.items() if isinstance(it, Curve) and it.closed]
        assert len(closed) == 1                      # only the diamond
        # the 2-point bar is gone: main curve + diamond, nothing else stroked
        assert [len(it.pts) for it in e.items() if isinstance(it, Curve)] == [2, 4]

    def test_an_untruncated_edge_is_unchanged(self):
        e = sch.edge((0.0, 0.0), (10.0, 0.0), "map")
        assert e.head_fractions() == (1.0,)
        assert not any(isinstance(it, MathLabel) for it in e.items())
        assert len(e.items()) == 1

    def test_a_truncated_edge_keeps_its_own_label(self):
        e = sch.edge((0.0, 0.0), (10.0, 0.0), "map", truncated=True, label="f")
        latexes = [it.latex for it in e.items() if isinstance(it, MathLabel)]
        assert "f" in latexes and any(r"\cdots" in x for x in latexes)


class TestOperatorJunction:
    def test_junction_glyph_and_arg_roles(self):
        j = sch.Junction("qk", (2.0, 3.0), r"\mathrm{QK}", radius=0.6,
                         args=((90.0, "q"), (180.0, "k")))
        labels = [it for it in j.items() if isinstance(it, MathLabel)]

        glyph = next(lb for lb in labels if lb.latex == r"\mathrm{QK}")
        assert glyph.anchor == pytest.approx((2.0, 3.0))
        assert glyph.ha == "center" and glyph.va == "center"

        q = next(lb for lb in labels if lb.latex == "q")
        assert q.anchor == pytest.approx((2.0, 3.0 + 1.45 * 0.6), abs=1e-9)
        k = next(lb for lb in labels if lb.latex == "k")
        assert k.anchor == pytest.approx((2.0 - 1.45 * 0.6, 3.0), abs=1e-9)

    def test_the_circle_is_drawn_at_the_stated_radius(self):
        j = sch.Junction("j", (1.0, -2.0), r"\odot", radius=0.35)
        (circle,) = [it for it in j.items() if isinstance(it, Curve)]

        assert circle.closed
        r = np.hypot(*(circle.pts - np.array([1.0, -2.0])).T)
        assert np.allclose(r, 0.35, atol=1e-12)

    def test_ports_lie_on_the_circle_at_the_stated_angle(self):
        j = sch.Junction("j", (2.0, 3.0), r"+", radius=0.6)

        assert j.port(0.0) == pytest.approx((2.6, 3.0))
        assert j.port(90.0) == pytest.approx((2.0, 3.6))
        assert j.port(180.0) == pytest.approx((1.4, 3.0))
        # a diagonal port is on the CIRCLE, not on a circumscribed square
        assert j.port(45.0) == pytest.approx(
            (2.0 + 0.6 / np.sqrt(2.0), 3.0 + 0.6 / np.sqrt(2.0)))
        assert j.boundary_distance(j.port(45.0)) < 1e-12

    def test_arg_labels_grow_away_from_the_circle(self):
        # ha/va are chosen per octant so the text never runs back over the
        # glyph: right half anchors on its left edge, top half on its bottom.
        j = sch.Junction("j", (0.0, 0.0), r"f", radius=1.0,
                         args=((0.0, "e"), (90.0, "n"), (180.0, "w"),
                               (270.0, "s"), (45.0, "ne")))
        by = {lb.latex: lb for lb in j.items() if isinstance(lb, MathLabel)}

        assert (by["e"].ha, by["e"].va) == ("left", "center")
        assert (by["w"].ha, by["w"].va) == ("right", "center")
        assert (by["n"].ha, by["n"].va) == ("center", "bottom")
        assert (by["s"].ha, by["s"].va) == ("center", "top")
        assert (by["ne"].ha, by["ne"].va) == ("left", "bottom")

    def test_arg_labels_are_annotation_and_sized_apart_from_the_glyph(self):
        j = sch.Junction("j", (0.0, 0.0), r"\odot", radius=0.5,
                         args=((90.0, "q"),), label_size_pt=13.0,
                         arg_size_pt=8.5, role=Role.ACCENT1)
        by = {lb.latex: lb for lb in j.items() if isinstance(lb, MathLabel)}

        assert by[r"\odot"].role is Role.ACCENT1
        assert by[r"\odot"].size_pt == 13.0
        assert by["q"].role is Role.ANNOTATION
        assert by["q"].size_pt == 8.5

    def test_a_filled_junction_masks_what_runs_behind_it(self):
        j = sch.Junction("j", (0.0, 0.0), r"+", radius=0.4, fill="#ffffff")
        its = j.items()

        (fill,) = [it for it in its if isinstance(it, FilledCurve)]
        assert fill.color == "#ffffff" and fill.opacity == 1.0
        assert fill.outline                     # the circle still reads
        assert not [it for it in its if isinstance(it, Curve)
                    and not isinstance(it, FilledCurve)]

    def test_an_unfilled_junction_is_one_stroked_circle(self):
        j = sch.Junction("j", (0.0, 0.0), r"+", radius=0.4)
        assert not [it for it in j.items() if isinstance(it, FilledCurve)]

    def test_a_junction_with_no_args_emits_only_circle_and_glyph(self):
        assert len(sch.Junction("j", (0.0, 0.0), r"+", radius=0.4).items()) == 2

    def test_a_junction_is_an_obstruction_like_a_node(self):
        j = sch.Junction("qk", (0.0, 0.0), r"\odot", radius=0.5)
        through = sch.edge((-3.0, 0.0), (3.0, 0.0), "map")
        arriving = sch.edge((-3.0, 0.0), j.port(180.0), "map")

        assert sch.clearance_violations([through], [j])
        assert sch.clearance_violations([arriving], [j]) == []

    def test_items_flattens_a_junction_like_a_node(self):
        j = sch.Junction("j", (0.0, 0.0), r"+", radius=0.4, args=((90.0, "q"),))
        flat = sch.items([j])
        assert [type(it) for it in flat] == [type(it) for it in j.items()]
        assert [it.latex for it in flat if isinstance(it, MathLabel)] == ["+", "q"]
