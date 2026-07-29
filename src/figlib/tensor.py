"""Tensor networks: an einsum drawn as a diagram, and gated as one.

Penrose / graphical-tensor notation (arXiv 2402.01790, arXiv 2102.13196):
**a node per tensor, a line per index, a joined line per contraction, a
dangling line per free axis.** Nothing is drawn to scale — a rank-3 tensor
has no 2-D silhouette — so the only geometry here is incidence.

**Why this is not `matrix.py`.** That module makes *shape* geometric: a
`Block` is drawn at its own aspect ratio, which is what makes a
non-conformable product undrawable. A tensor's structure is which axes are
identified with which, which is a graph. Giving it a rectangle would assert
a shape it does not have.

**Why it is a module at all.** The test from `primitive-gaps.md` — not "is
this device inexpressible" but "does this make a new class of claim
checkable". Three claims become checkable, and none can live inside one
figure because each needs the incidence structure and the arrays to be one
object surviving `compute()` -> `assertions()`:

1. `spec()` derives the einsum string FROM the picture and `contract()`
   evaluates it, so a printed einsum string is read off the drawing.
2. Two different diagrams are the same contraction (`check_einsum` on
   both) — the mech-interp claim that only `W_Q W_K^T` matters.
3. Every contracted index has one dimension (`check_index_dims`) — the
   conformability analogue.

**Geometry is delegated to `schematic`.** A tensor emits a `schematic.Node`
and every line is a `schematic.Edge`, so `clearance_violations`,
`crossing_count` and `assemble`'s draw order apply unchanged. There is no
renderer here and no second shape model.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from string import ascii_lowercase
from typing import Sequence

import numpy as np

from .scene import Item
from .schematic import (EDGE_KINDS, Edge, EdgeDecor, Node, assemble,
                        circle_node, edge)
from .style import Role

XY = tuple[float, float]

# All five existing edge kinds are directed: they say something FLOWS. A
# contraction is an identification of two axes and has no direction, so an
# arrowhead on it would assert a mechanism the algebra does not have. That
# is the distinction `schematic.EDGE_KINDS` asks a sixth kind to earn.
EDGE_KINDS.setdefault("wire", EdgeDecor(head="none", chevrons=0))


@dataclass(frozen=True)
class Leg:
    """One index of one tensor, leaving the node at a stated angle.

    No `dim` field. Dimensions are DERIVED from `Tensor.values.shape`; a
    stated dim would be a second source of truth, and `check_index_dims`
    would then be checking the figure against itself.
    """

    index: str
    angle: float                       # degrees CCW from +x
    length: float | None = None        # None -> the network's default stub
    label: str | None = None           # None -> the index name


@dataclass(frozen=True)
class Tensor:
    """A node in the network. `legs` ORDER IS THE AXIS ORDER of `values`."""

    key: str
    center: XY
    legs: tuple[Leg, ...]
    values: np.ndarray | None = None
    label: str | None = None
    radius: float = 0.28               # circle radius, or half-height of a box
    shape: str = "circle"              # circle | box
    width: float | None = None         # box only; None -> 2 * radius
    role: Role = Role.CONTENT
    fill: str | None = None            # paper colour, so wires pass behind
    label_size_pt: float | None = None

    @property
    def half(self) -> XY:
        w = self.width if (self.shape == "box" and self.width) else 2 * self.radius
        return (0.5 * float(w), float(self.radius))

    def port(self, leg: Leg) -> XY:
        """Where `leg` meets the boundary, along its own angle.

        Circle: the point on the circle. Box: the ray-rect intersection —
        so a leg on a diagonal lands on the ink either way.
        """
        th = np.radians(float(leg.angle))
        c, s = float(np.cos(th)), float(np.sin(th))
        cx, cy = self.center
        if self.shape == "circle":
            return (cx + self.radius * c, cy + self.radius * s)
        hw, hh = self.half
        # scale the ray until it first touches a side
        t = min(hw / abs(c) if abs(c) > 1e-12 else np.inf,
                hh / abs(s) if abs(s) > 1e-12 else np.inf)
        return (cx + t * c, cy + t * s)

    def node(self) -> Node:
        if self.shape == "circle":
            return circle_node(self.key, self.center, self.radius,
                               label=self.label, role=self.role,
                               fill=self.fill, label_size_pt=self.label_size_pt)
        hw, hh = self.half
        return Node(self.key, self.center, 2 * hw, 2 * hh, label=self.label,
                    role=self.role, fill=self.fill,
                    label_size_pt=self.label_size_pt)


@dataclass(frozen=True)
class Network:
    """Tensors plus the free-index order. Everything else is derived."""

    tensors: tuple[Tensor, ...]
    out: tuple[str, ...] | None = None      # free-index order; None -> sorted
    stub: float = 0.42                      # default free-leg length
    route: dict[str, str] = field(default_factory=dict)   # per-index override
    via: dict[str, XY] = field(default_factory=dict)
    label_size_pt: float | None = None
    index_role: Role = Role.ANNOTATION


# --- index bookkeeping ------------------------------------------------------


def _legs(net: Network):
    for t in net.tensors:
        for k, leg in enumerate(t.legs):
            yield t, k, leg


def arity(net: Network) -> dict[str, int]:
    """index -> how many legs carry it."""
    return dict(Counter(leg.index for _, _, leg in _legs(net)))


def contracted(net: Network) -> tuple[str, ...]:
    """Indices on exactly two legs, in first-appearance order."""
    a = arity(net)
    seen: list[str] = []
    for _, _, leg in _legs(net):
        if a[leg.index] == 2 and leg.index not in seen:
            seen.append(leg.index)
    return tuple(seen)


def free(net: Network) -> tuple[str, ...]:
    """Dangling indices — the output axes — in `net.out` order if stated."""
    a = arity(net)
    dangling = [i for i in dict.fromkeys(leg.index for _, _, leg in _legs(net))
                if a[i] == 1]
    if net.out is None:
        return tuple(sorted(dangling))
    stated, loose = list(net.out), set(dangling)
    for i in stated:
        if i not in loose:
            raise ValueError(
                f"out names {i!r}, which is contracted (arity {a.get(i, 0)}); "
                f"a contracted index is summed away and cannot be an output")
    missing = [i for i in dangling if i not in stated]
    if missing:
        raise ValueError(f"out omits dangling indices {missing}; every free "
                         f"leg is an output axis")
    return tuple(stated)


def dims(net: Network) -> dict[str, int]:
    """index -> length, read off the arrays. Requires every tensor to carry
    `values`: a dimension inferred from nothing is not checkable."""
    out: dict[str, int] = {}
    for t in net.tensors:
        if t.values is None:
            raise ValueError(f"tensor {t.key!r} has no values; dims are "
                             f"derived from the arrays, never stated")
        v = np.asarray(t.values)
        if v.ndim != len(t.legs):
            raise ValueError(f"tensor {t.key!r} has {len(t.legs)} legs but a "
                             f"rank-{v.ndim} array; legs ARE the axes")
        for leg, n in zip(t.legs, v.shape):
            out.setdefault(leg.index, int(n))
    return out


def _letters(net: Network) -> dict[str, str]:
    """Index name -> a single einsum letter. Figures name indices `seq`,
    `d_model`; einsum wants one character, and the mapping has to be
    deterministic or the derived spec is not reproducible."""
    names = list(dict.fromkeys(leg.index for _, _, leg in _legs(net)))
    pool = [c for c in ascii_lowercase if c not in names]
    return {n: (n if len(n) == 1 else pool.pop(0)) for n in names}


def spec(net: Network) -> str:
    """The einsum string, read off the drawing."""
    L = _letters(net)
    lhs = ",".join("".join(L[leg.index] for leg in t.legs) for t in net.tensors)
    return f"{lhs}->{''.join(L[i] for i in free(net))}"


def contract(net: Network) -> np.ndarray:
    """Evaluate the drawn network. This is the whole point of the module:
    the picture and the `np.einsum` call are one object."""
    dims(net)                                     # raises on rank/value gaps
    return np.einsum(spec(net), *[np.asarray(t.values) for t in net.tensors])


# --- marks ------------------------------------------------------------------


def nodes(net: Network) -> list[Node]:
    return [t.node() for t in net.tensors]


def _tip(t: Tensor, leg: Leg, length: float) -> XY:
    p = t.port(leg)
    th = np.radians(float(leg.angle))
    return (p[0] + length * float(np.cos(th)), p[1] + length * float(np.sin(th)))


def _pairs(net: Network) -> dict[str, list[tuple[Tensor, Leg]]]:
    out: dict[str, list[tuple[Tensor, Leg]]] = {}
    for t, _, leg in _legs(net):
        out.setdefault(leg.index, []).append((t, leg))
    return out


def edges(net: Network) -> list[Edge]:
    """One `wire` per contraction, one dangling stub per free index.

    The index name is labelled ONCE per line — on a wire at mid-arc with a
    halo, on a stub at its tip — because the line is the index, and naming
    it twice would read as two different axes.
    """
    a = arity(net)
    out: list[Edge] = []
    for index, legs in _pairs(net).items():
        kw = dict(kind="wire", key=index, role=net.index_role,
                  label=legs[0][1].label or index,
                  label_size_pt=net.label_size_pt,
                  label_role=net.index_role)
        if a[index] == 2:
            (t0, l0), (t1, l1) = legs
            p, q = t0.port(l0), t1.port(l1)
            if t0 is t1:
                # a self-contraction (a trace) has no straight route: bow it
                # out past the node so the loop is visible as a loop
                cx, cy = t0.center
                mid = t0.port(Leg(index, 0.5 * (l0.angle + l1.angle)))
                r = 2.2 * max(t0.half)
                via = (cx + r * (mid[0] - cx) / max(t0.radius, 1e-9),
                       cy + r * (mid[1] - cy) / max(t0.radius, 1e-9))
                out.append(edge(p, q, route="quad", via=net.via.get(index, via),
                                label_halo=True, **kw))
                continue
            out.append(edge(p, q, route=net.route.get(index, "straight"),
                            via=net.via.get(index), label_halo=True, **kw))
        else:
            (t0, l0), = legs
            length = l0.length if l0.length is not None else net.stub
            out.append(edge(t0.port(l0), _tip(t0, l0, length),
                            label_at=1.0, label_va="center", **kw))
    return out


def items(net: Network, *, under: Sequence = (), over: Sequence = ()
          ) -> list[Item]:
    """Scene items in `assemble`'s order: wires down first, paper-filled
    nodes over them, index labels back on top."""
    return assemble(nodes(net), edges(net), under=under, over=over)


def extent(net: Network, pad: float = 0.0) -> tuple[float, float, float, float]:
    """(x0, x1, y0, y1) over nodes AND stubs — a dangling leg is content,
    and a bound taken over the nodes alone clips every output axis."""
    xs: list[float] = []
    ys: list[float] = []
    for t in net.tensors:
        hw, hh = t.half
        xs += [t.center[0] - hw, t.center[0] + hw]
        ys += [t.center[1] - hh, t.center[1] + hh]
    for e in edges(net):
        xs += [float(e.pts[:, 0].min()), float(e.pts[:, 0].max())]
        ys += [float(e.pts[:, 1].min()), float(e.pts[:, 1].max())]
    return (min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad)


# --- gates ------------------------------------------------------------------


def check_index_arity(c, net: Network) -> None:
    """A line has two ends. An index on three or more legs is a legal
    einsum and not a drawable network — the diagram would be asserting a
    pairwise identification that is not what the sum does."""
    for index, n in arity(net).items():
        c.check(n <= 2,
                f"index {index!r} is on {n} legs; a line has two ends, so "
                f"this network cannot be drawn — split the tensor or use an "
                f"explicit copy node")


def check_index_dims(c, net: Network) -> None:
    """Both axes joined by a wire have the same length. The conformability
    analogue: this is what makes a wrong wire fail rather than look fine."""
    seen: dict[str, tuple[str, int]] = {}
    for t in net.tensors:
        v = np.asarray(t.values)
        for leg, n in zip(t.legs, v.shape):
            prev = seen.get(leg.index)
            if prev is None:
                seen[leg.index] = (t.key, int(n))
            else:
                c.check(prev[1] == int(n),
                        f"index {leg.index!r} joins {prev[0]} axis of length "
                        f"{prev[1]} to {t.key} axis of length {n} — the wire "
                        f"is drawn but the contraction does not exist")


def check_einsum(c, net: Network, expected, *, rtol: float = 1e-9,
                 atol: float = 0.0) -> None:
    """The DRAWN network evaluates to `expected`."""
    try:
        got = contract(net)
    except Exception as exc:                     # a spec that will not run
        c.check(False, f"the drawn network does not contract: {exc}")
        return
    exp = np.asarray(expected)
    c.check(got.shape == exp.shape,
            f"the drawn network contracts to shape {got.shape}, not "
            f"{exp.shape} (einsum {spec(net)!r})")
    if got.shape == exp.shape:
        err = float(np.max(np.abs(got - exp))) if got.size else 0.0
        c.check(np.allclose(got, exp, rtol=rtol, atol=atol),
                f"the drawn network ({spec(net)!r}) does not evaluate to the "
                f"claimed array: max |diff| = {err:.3e}")


def check_output(c, net: Network, shape: Sequence[int]) -> None:
    """The dangling legs are the output axes, in the stated order."""
    d = dims(net)
    got = tuple(d[i] for i in free(net))
    c.check(got == tuple(int(s) for s in shape),
            f"the dangling legs {free(net)} have dims {got}, but the figure "
            f"claims output shape {tuple(shape)}")


def check_leg_aim(c, net: Network, tol_deg: float = 75.0) -> None:
    """A contracted leg leaves in roughly the direction of its partner.

    A leg pointing the wrong way still renders — the wire is drawn between
    the two ports whatever they are — but it leaves the node on the far
    side and grazes its own body, so the reader attaches it to the wrong
    axis. Purely a drawing fact, which is why it is a gate and not a
    computation: nothing upstream can catch it.
    """
    for index, legs in _pairs(net).items():
        if len(legs) != 2:
            continue
        (t0, l0), (t1, l1) = legs
        if t0 is t1:
            continue
        for (ta, la), (tb, lb) in ((legs[0], legs[1]), (legs[1], legs[0])):
            p, q = np.array(ta.port(la)), np.array(tb.port(lb))
            bearing = np.degrees(np.arctan2(q[1] - p[1], q[0] - p[0]))
            err = abs((float(la.angle) - bearing + 180.0) % 360.0 - 180.0)
            c.check(err <= tol_deg,
                    f"index {index!r}: {ta.key}'s leg leaves at "
                    f"{la.angle:g} deg but {tb.key} is at {bearing:.0f} deg "
                    f"({err:.0f} deg off) — the wire will graze {ta.key} and "
                    f"read as attached to the wrong side")
