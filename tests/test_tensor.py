"""Tensor networks: index bookkeeping, derived einsum, marks, gates."""

import numpy as np
import pytest

from figlib.gates import Checks
from figlib.scene import Curve, FilledCurve, MathLabel
from figlib.schematic import EDGE_KINDS, Edge, Node, crossing_count
from figlib.style import Role
from figlib.tensor import (Leg, Network, Tensor, arity, check_einsum,
                           check_index_arity, check_index_dims,
                           check_leg_aim, check_output, contract, contracted,
                           dims, edges, extent, free, items, nodes, spec)


def _mat(key, center, i, j, values, **kw):
    """A rank-2 tensor with its two legs left (i) and right (j)."""
    return Tensor(key, center, (Leg(i, 180.0), Leg(j, 0.0)),
                  values=values, label=key, **kw)


def _chain():
    """s-[A]-d-[B]-t : a plain matrix product, drawn."""
    A = np.arange(6.0).reshape(2, 3)
    B = np.arange(12.0).reshape(3, 4)
    return Network((_mat("A", (0.0, 0.0), "s", "d", A),
                    _mat("B", (1.6, 0.0), "d", "t", B)), out=("s", "t"))


# --- index bookkeeping ---------------------------------------------------

def test_arity_counts_every_leg():
    assert arity(_chain()) == {"s": 1, "d": 2, "t": 1}


def test_free_and_contracted_split_by_arity():
    n = _chain()
    assert free(n) == ("s", "t")
    assert contracted(n) == ("d",)


def test_free_order_follows_out_when_stated():
    A = np.arange(6.0).reshape(2, 3)
    B = np.arange(12.0).reshape(3, 4)
    n = Network((_mat("A", (0.0, 0.0), "s", "d", A),
                 _mat("B", (1.6, 0.0), "d", "t", B)), out=("t", "s"))
    assert free(n) == ("t", "s")


def test_out_naming_an_index_that_is_contracted_is_an_error():
    with pytest.raises(ValueError, match="contracted"):
        free(Network(_chain().tensors, out=("d",)))


def test_dims_come_from_the_arrays_not_from_the_legs():
    assert dims(_chain()) == {"s": 2, "d": 3, "t": 4}


def test_dims_needs_values():
    n = Network((Tensor("A", (0.0, 0.0), (Leg("i", 0.0),)),))
    with pytest.raises(ValueError, match="values"):
        dims(n)


def test_leg_count_must_match_array_rank():
    with pytest.raises(ValueError, match="legs"):
        dims(Network((Tensor("A", (0.0, 0.0), (Leg("i", 0.0),),
                             values=np.zeros((2, 3))),)))


# --- the einsum derived FROM the drawing ---------------------------------

def test_spec_is_read_off_the_diagram():
    assert spec(_chain()) == "sd,dt->st"


def test_spec_respects_leg_order_as_axis_order():
    A = np.arange(6.0).reshape(2, 3)
    B = np.arange(12.0).reshape(3, 4)
    # B drawn with its legs listed the other way round IS B transposed
    Bt = Tensor("B", (1.6, 0.0), (Leg("t", 0.0), Leg("d", 180.0)),
                values=B.T, label="B")
    n = Network((_mat("A", (0.0, 0.0), "s", "d", A), Bt), out=("s", "t"))
    assert spec(n) == "sd,td->st"
    assert np.allclose(contract(n), A @ B)


def test_contract_matches_the_matrix_product():
    n = _chain()
    A, B = n.tensors[0].values, n.tensors[1].values
    assert np.allclose(contract(n), A @ B)


def test_multi_letter_index_names_are_mapped_to_single_letters():
    A = np.arange(6.0).reshape(2, 3)
    B = np.arange(12.0).reshape(3, 4)
    n = Network((_mat("A", (0.0, 0.0), "seq", "model", A),
                 _mat("B", (1.6, 0.0), "model", "head", B)),
                out=("seq", "head"))
    s = spec(n)
    assert len(s.split("->")[1]) == 2      # one letter per free index
    assert np.allclose(contract(n), A @ B)


def test_trace_is_a_tensor_wired_to_itself():
    M = np.arange(9.0).reshape(3, 3)
    n = Network((Tensor("M", (0.0, 0.0), (Leg("i", 90.0), Leg("i", 270.0)),
                        values=M, label="M"),), out=())
    assert spec(n) == "ii->"
    assert np.isclose(float(contract(n)), np.trace(M))


# --- marks ---------------------------------------------------------------

def test_a_tensor_becomes_one_node():
    ns = nodes(_chain())
    assert [nd.key for nd in ns] == ["A", "B"]
    assert all(isinstance(nd, Node) for nd in ns)
    # round by default: a circle is width == height with the radius maxed
    assert ns[0].width == ns[0].height == pytest.approx(2 * ns[0].radius)


def test_box_shape_is_not_round():
    t = Tensor("W", (0.0, 0.0), (Leg("i", 0.0),), values=np.zeros(3),
               label="W", shape="box")
    nd = nodes(Network((t,), out=("i",)))[0]
    assert nd.radius < 0.5 * min(nd.width, nd.height)


def test_a_contraction_is_one_undirected_wire_between_the_two_ports():
    es = edges(_chain())
    wires = [e for e in es if e.key == "d"]
    assert len(wires) == 1
    w = wires[0]
    assert w.kind == "wire"
    assert EDGE_KINDS["wire"].head == "none"
    assert w.items()[0].arrows == ()          # no head: no direction to assert
    # it runs between the two tensors, left port of B and right port of A
    assert w.anchors[0][0] < w.anchors[1][0]


def test_a_free_index_is_a_dangling_stub_of_the_stated_length():
    n = _chain()
    stubs = [e for e in edges(n) if e.key == "s"]
    assert len(stubs) == 1
    p, q = np.array(stubs[0].anchors[0]), np.array(stubs[0].anchors[1])
    assert float(np.hypot(*(q - p))) == pytest.approx(n.stub)
    # leg angle 180 deg: the stub leaves to the LEFT
    assert q[0] < p[0] and abs(q[1] - p[1]) < 1e-12


def test_every_index_is_labelled_exactly_once():
    labels = [it.latex for it in items(_chain()) if isinstance(it, MathLabel)]
    for idx in ("s", "d", "t"):
        assert labels.count(idx) == 1


def test_items_draws_wires_under_the_nodes():
    """A wire painted OVER a paper-filled tensor would run through its
    symbol; assemble()'s order is what makes the node mask its own wires."""
    its = items(_chain())
    wires = [k for k, it in enumerate(its)
             if isinstance(it, Curve) and not it.closed]
    bodies = [k for k, it in enumerate(its)
              if isinstance(it, FilledCurve)
              or (isinstance(it, Curve) and it.closed)]
    assert wires and bodies
    assert max(wires) < min(bodies)


def test_extent_covers_the_stubs_not_just_the_nodes():
    n = _chain()
    x0, x1, y0, y1 = extent(n, pad=0.0)
    assert x0 == pytest.approx(-n.tensors[0].radius - n.stub)
    assert x1 == pytest.approx(1.6 + n.tensors[1].radius + n.stub)


# --- gates: each fires on a violation and is silent on a valid network ----

def test_index_arity_gate_is_silent_on_a_drawable_network():
    c = Checks()
    check_index_arity(c, _chain())
    c.done()


def test_index_arity_gate_fires_on_a_three_legged_index():
    A = np.zeros((2, 3))
    n = Network((_mat("A", (0.0, 0.0), "s", "d", A),
                 _mat("B", (1.6, 0.0), "d", "t", np.zeros((3, 4))),
                 _mat("C", (3.2, 0.0), "d", "u", np.zeros((3, 5)))))
    c = Checks()
    check_index_arity(c, n)
    with pytest.raises(AssertionError, match="d.*3"):
        c.done()


def test_index_dims_gate_fires_on_a_mismatched_contraction():
    n = Network((_mat("A", (0.0, 0.0), "s", "d", np.zeros((2, 3))),
                 _mat("B", (1.6, 0.0), "d", "t", np.zeros((5, 4)))))
    c = Checks()
    check_index_dims(c, n)
    with pytest.raises(AssertionError, match="d"):
        c.done()


def test_index_dims_gate_is_silent_when_the_wire_is_conformable():
    c = Checks()
    check_index_dims(c, _chain())
    c.done()


def test_einsum_gate_confirms_the_drawn_network():
    n = _chain()
    A, B = n.tensors[0].values, n.tensors[1].values
    c = Checks()
    check_einsum(c, n, A @ B)
    c.done()


def test_einsum_gate_fires_when_the_drawing_means_something_else():
    n = _chain()
    A, B = n.tensors[0].values, n.tensors[1].values
    c = Checks()
    check_einsum(c, n, (A @ B) + 1.0)
    with pytest.raises(AssertionError, match="does not evaluate"):
        c.done()


def test_output_gate_checks_the_dangling_legs():
    c = Checks()
    check_output(c, _chain(), (2, 4))
    c.done()


def test_output_gate_fires_on_the_wrong_output_shape():
    c = Checks()
    check_output(c, _chain(), (4, 2))
    with pytest.raises(AssertionError, match="output"):
        c.done()


def test_leg_aim_gate_is_silent_when_legs_face_their_partner():
    c = Checks()
    check_leg_aim(c, _chain())
    c.done()


def test_leg_aim_gate_fires_when_a_leg_points_away_from_its_partner():
    # A's "d" leg leaves to the LEFT while B sits to the right of it
    A = Tensor("A", (0.0, 0.0), (Leg("s", 90.0), Leg("d", 180.0)),
               values=np.zeros((2, 3)), label="A")
    n = Network((A, _mat("B", (1.6, 0.0), "d", "t", np.zeros((3, 4)))))
    c = Checks()
    check_leg_aim(c, n)
    with pytest.raises(AssertionError, match="d"):
        c.done()


def test_the_schematic_checks_still_apply_to_a_network():
    # crossing_count is schematic's, and it must accept tensor edges unchanged
    assert crossing_count(edges(_chain())) == 0
    assert all(isinstance(e, Edge) for e in edges(_chain()))
