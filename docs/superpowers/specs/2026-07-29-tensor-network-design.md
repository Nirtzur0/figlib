# Tensor networks (einsum diagrams) — design

**Status:** spec. Closes the item deferred by the matrix-layer spec
(`2026-07-29-matrix-layer-design.md` §"Out of scope") and by
`primitive-gaps.md` ("Still open, deliberately deferred").

**Goal.** Draw an einsum as a diagram whose *contraction is gated*: the
picture and the `np.einsum` call are one object, so a diagram that means a
different contraction than the caption claims cannot render.

## 0. Why this is a module, and why it is not `matrix.py`

`matrix.py` makes **shape geometric** — a `Block` is drawn at its own
aspect ratio, so a non-conformable product is undrawable. A tensor of rank
3+ has no 2-D shape to draw to scale. Its structure is *which axes are
identified with which*, which is a graph, not a rectangle. Forcing it into
`Block` would corrupt the shape-is-geometry invariant; giving it its own
2-D silhouette would assert a shape the tensor does not have.

So the object here is Penrose/graphical-tensor notation (arXiv 2402.01790,
arXiv 2102.13196): **a node per tensor, a line per index, a joined line per
contraction, a dangling line per free axis.** Nothing is drawn to scale;
the only geometry is incidence.

Applying the module test from `primitive-gaps.md` — *not "is this device
inexpressible" but "does this make a new class of claim checkable"* — the
new checkable claims are:

1. **The drawn diagram contracts to this array.** `spec(net)` derives the
   einsum string *from the picture* and `contract(net)` evaluates it. A
   figure that prints its einsum string prints a value read off the
   drawing, so caption and diagram cannot drift apart.
2. **Two different diagrams are the same contraction.** The mech-interp
   claim (`W_Q W_K^T` is one bilinear form; only the product matters)
   becomes `np.allclose(contract(a), contract(b))` on two drawn networks.
3. **Every contracted index has one dimension.** The conformability
   analogue, and the reason a wrong wire fails a gate rather than looking
   fine.

None of these can live inside one figure: each needs the incidence
structure and the arrays to be one object surviving `compute()` →
`assertions()`.

## 1. It rides on `schematic.py`

`tensor.py` owns index bookkeeping and gates. Geometry is delegated:
tensors emit `schematic.Node` (round, via `circle_node`) or a box, and
every line is a `schematic.Edge`, so `clearance_violations`,
`crossing_count`, `assemble` and the draw-order rules apply unchanged. No
new renderer, no second shape model.

One addition to `schematic.EDGE_KINDS`:

```python
"wire": EdgeDecor(head="none", chevrons=0),
```

justified by the module's own rule ("adding a sixth kind should require a
mechanism none of these expresses"): all five existing kinds are directed —
they say something flows. A contraction is an **identification of two
axes**; it has no direction, and drawing an arrowhead on it would assert a
flow the algebra does not have.

## 2. The objects

```python
@dataclass(frozen=True)
class Leg:
    index: str                 # the einsum index name
    angle: float               # degrees CCW from +x: where it leaves the node
    length: float | None = None    # None -> the network's default stub length
    label: str | None = None       # None -> the index name

@dataclass(frozen=True)
class Tensor:
    key: str
    center: XY
    legs: tuple[Leg, ...]      # ORDER IS THE AXIS ORDER of `values`
    values: np.ndarray | None = None
    label: str | None = None
    radius: float = 0.28
    shape: str = "circle"      # circle | box
    ...role/fill/label styling
```

**Dimensions are never stated, only derived** from `values.shape` aligned
with `legs`. A stated `dim` would be a second source of truth that can
disagree with the array; `check_index_dims` would then be checking the
figure against itself.

```python
@dataclass(frozen=True)
class Network:
    tensors: tuple[Tensor, ...]
    out: tuple[str, ...] | None = None   # free-index order; None -> sorted
    stub: float = 0.42                   # default free-leg / half-wire length
    route: dict[str, str] = ...          # per-index routing override
    via: dict[str, XY] = ...
```

Derived, all pure functions of the above:

- `arity(net)` — index -> occurrence count
- `free(net)`, `contracted(net)`
- `dims(net)` — index -> int, from the arrays
- `spec(net)` -> `"sd,dh,te,eh->st"`
- `contract(net)` -> `np.einsum(spec, *values)`
- `items(net)` -> scene items (nodes, contraction wires, free stubs, labels)

## 3. Marks

- **Tensor** — a small circle with its symbol inside, paper-filled so wires
  pass behind it. A box when the tensor is a named block (learned weights);
  `shape="box"` is the only styling choice that carries meaning here.
- **Contraction** — one `wire` edge from port to port, no head, index name
  labelled once at mid-arc with a halo.
- **Free leg** — a stub of `stub` length leaving the port along the leg
  angle, index name at the tip. A dangling line *is* an output axis; that
  is the whole reason the notation works.
- **No dimension text on the wires by default.** The dims are gated, not
  printed; a figure that wants them prints them from `dims(net)` so they
  cannot go stale.

## 4. Gates

| gate | what could be wrong |
|---|---|
| `check_index_arity` | an index on 3+ legs — an einsum that is not a drawable network (a line has two ends) |
| `check_index_dims` | a contracted index whose two axes have different lengths — the conformability analogue |
| `check_einsum(c, net, expected)` | the **drawn** network evaluates to the array the figure claims |
| `check_output(c, net, shape)` | the dangling legs are the output axes, in the stated order |
| `check_leg_aim(c, net, tol_deg)` | a leg leaves at an angle pointing away from its partner — the wire then grazes its own node and reads as attached to the wrong side |
| (reuse) `clearance_violations`, `crossing_count` | routing honesty, already owned by `schematic` |

Not gated, deliberately: that the picture "looks like" a tensor network.
There is nothing to check — incidence is the content and it is checked
above.

## 5. Benchmark figure

`figures/qk_circuit_tensor.py` — the QK circuit of one attention head, twice:

```
row 1:  s—[X]—d—[W_Q]—h—[W_K]—e—[X]—t        spec "sd,dh,te,eh->st"
row 2:  s—[X]—d—[W_QK]—e—[X]—t               spec "sd,de,te->st"
```

The claim is that these are the same array — `W_Q W_K^T` is one bilinear
form of rank ≤ d_head, so the factorization into query and key spaces is a
parameterization, not a mechanism. Assertions: both networks contract to
the same array (`np.allclose`), `rank(W_QK) == d_head < d_model`, both
specs derived from the drawings, index arity and dims on both.

That figure is the reason for the module: the claim is a statement about
*two diagrams*, which no amount of careful drawing can establish and one
`np.allclose` does.
