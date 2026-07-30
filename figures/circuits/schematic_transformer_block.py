"""The pre-norm transformer block, drawn as a residual stream with branches.

Design steps (architecture.md 0-9), recorded because for a Class B figure
the layout IS the content and there is no numerics to fall back on.

0. EARN IT. The prose — "each sublayer computes a function of the
   normalized stream and adds its output back" — is true, short, and
   installs the wrong picture: read linearly it sounds like a pipeline,
   x -> LN -> Attn -> LN -> MLP -> y, with each stage consuming the last.
   The inference the reader has to make is that nothing on that list is on
   the main path at all. On the page that becomes perceptual, because the
   main path is drawn as a path: one heavy line the eye follows from
   bottom to top without ever entering a box.
1. CLAIM. Below.
2. REPRESENTATION. The residual stream as the SPINE — the
   mechanistic-interpretability reading, where the stream is the object
   and the sublayers are things that read from and write to it. Rung: one
   instance (a single block), not a stack, because the claim is about what
   is on the main path and a stack would only repeat it. The expert's
   private picture is a bus with taps, and that is what gets drawn:
   junction dots where a sublayer reads, hollow heads where it writes back.
   The two branches go to OPPOSITE sides. That was not the first attempt —
   both on the right left half the column empty and made the taps travel
   300 px over nothing — and it is not a claim of parallelism: the ranks
   still order them, and each branch returns to its own +. What it buys is
   that the spine sits dead center, which is where the figure's subject
   belongs.
3. SIZE. COLUMN. The whole layout is stated in px at this format (see
   PARAMS) and converted once through `px_per_unit`, so the box padding,
   the rank spacing and the tap length mean the same thing on the page
   that they mean in the source. Eight ranks make it taller than it is
   wide; that is what a residual stream drawn vertically costs.
4. TRAVERSAL. Enter at the bottom on the heaviest ink and ride it up: the
   eye is on the residual stream before it knows what a LayerNorm is. The
   forks read as departures FROM that line, and each returns to it at a
   circled +. A 3-second glance yields "one thick line up the middle, two
   loops off it"; a 30-second read yields the sublayer order and the fact
   that the norms are inside the loops.
5. MECHANISM ANNOTATION. Readable off the figure: that the LayerNorms are
   inside the branches and never on the stream (pre-norm); that the stream
   and every write are d_model wide, which is WHY the add is defined; that
   each branch reads the stream strictly below where it writes back.
6. READER EFFORT. Delegated: composing "the branch reads at the dot" with
   "the branch writes at the +" into "the stream survives the block
   additively". Kept: nothing about decoding — flow direction is on every
   arrowhead.
7. CHANNELS. Weight is the hierarchy, not hue: the stream is CONTENT at
   2.2x, the branches CONTENT at 1.0x, and the figure needs no accent
   because it distinguishes one object, not two. Kind = relation: `excite`
   (filled head) where the stream carries itself forward, `map` (hollow
   head) where a sublayer transforms what it read. Boxes are paper-filled
   so the spine and the taps pass cleanly behind them. CONTAINMENT is the
   fourth channel: each norm+sublayer pair sits on a grouping ground, so
   "one sublayer" is a thing the eye gets for free rather than an inference
   from adjacency. The transformer-circuits corpus draws that ground as a
   bare wash at about 1.03:1 and no border. That does not survive here —
   the house fill floor is 1.3:1, and a compliant wash over an area this
   large stops being background and becomes the heaviest ink on the page
   (measured: 59% of ink at REGION_OPACITY). So the ground is drawn the
   way this theme can afford it: a CONSTRUCTION-dashed border, which the
   colour gate exempts because the line is then the mark carrying the
   contrast, over a wash at 0.10 that only has to be findable. Louder than
   the corpus, and the corpus value is unavailable on grained paper.
8. HONESTY. Two of the elisions are now DRAWN instead of confessed here.
   The attention box carries `stack=2` ghost cards: one box abbreviating
   n heads, said on the page. And the stream above x_{l+1} continues on a
   `truncated` edge — hollow diamond, ellipsis — because the block
   repeats and the subscript alone was carrying that silently. Still
   absent and still only confessed: the QKV projections, the softmax, the
   MLP's inner width and nonlinearity, dropout, and the final LayerNorm a
   pre-norm stack needs at the top. It also draws the attention sublayer
   as if it saw one position, when the whole point of attention is that it
   reads OTHER positions' streams — this is a single column for a single
   token, and that is the figure's main lie. No QK junction and no mono
   token strip, deliberately: the figure's claim is about what is on the
   residual stream, the attention box is a single abbreviated node (that
   is what the stack says), and nothing drawn here is literal model input
   — x_l is a mid-stack activation, not a token.
9. GATES. Numerical = the structural asserts below: the drawn y-order
   agrees with the DAG's longest-path ranking (no edge runs backwards),
   NO SUBLAYER GROUND straddles the spine lane (the pre-norm claim, made
   of the whole sublayer rather than of the norm alone — a later edit that
   widened a box could not sneak across), each ground holds the two boxes
   it claims and the grounds nest-or-are-disjoint, the two branches are on
   opposite sides, both adds take exactly two inputs, ports are exact,
   zero crossings, no edge pierces a box it is not attached to.
"""

import numpy as np

from figlib import schematic as sch
from figlib.format import COLUMN
from figlib.scene import MathLabel, Point, Scene
from figlib.style import Role
from figlib.theme import RISO

CLAIM = (
    "In a pre-norm transformer block the residual stream is the main path and "
    "nothing sits on it: attention and the MLP are branches that READ a "
    "normalized copy of the stream and ADD their output back, so the "
    "LayerNorms never touch what is carried forward and every write is the "
    "same d_model width as the stream it is added to."
)

EXPOSITION = """
Elhage et al., "A Mathematical Framework for Transformer Circuits"
(transformer-circuits.pub, 2021), open by discarding the layer diagram
everyone draws first. A transformer is not a stack of stages each
consuming the last; it is a residual stream -- a d_model-wide vector
per token, read and written additively by every attention head and MLP
in the network -- with each sublayer's output a small perturbation on
top of an otherwise-untouched channel. That reframing is what makes the
rest of the paper's math work: because every layer's contribution is
additive, the network's output decomposes into a sum over paths through
the stream, and a path can be analyzed in isolation from every other
one. The framework's central objects -- QK and OV circuits, induction
heads, virtual attention heads formed by composing two real ones -- are
all statements about what gets written to the stream and what a later
layer reads back out of it, and none of them are visible in the
box-and-arrow pipeline picture. This figure draws the one fact the
prose states but the pipeline diagram hides: LayerNorm is inside the
branch, not on the spine, so what the next sublayer reads is a
normalized copy while what survives to the end of the network is the
raw sum. Getting that ordering backwards -- treating the normalized
stream as what persists -- silently breaks the additive decomposition
the whole framework depends on.
"""

THEME = RISO
FORMAT = COLUMN

PARAMS = {
    # the graph, stated once; ranks are the LONGEST path through it, which
    # is what puts each + one rank above the sublayer feeding it instead of
    # beside the node it skips from
    "dag": [("in", "ln1"), ("ln1", "attn"), ("attn", "add1"), ("in", "add1"),
            ("add1", "ln2"), ("ln2", "mlp"), ("mlp", "add2"),
            ("add1", "add2"), ("add2", "out")],
    "spine": ["in", "add1", "add2", "out"],
    # (LayerNorm, sublayer, the + it writes to, which side of the stream)
    "branches": [("ln1", "attn", "add1", -1), ("ln2", "mlp", "add2", +1)],
    "labels": {
        "in": r"\mathbf{x}_{\ell}",
        "ln1": r"\text{LayerNorm}",
        "attn": r"\text{Multi-Head Attention}",
        "ln2": r"\text{LayerNorm}",
        "mlp": r"\text{MLP}",
        "out": r"\mathbf{x}_{\ell+1}",
    },
    # Layout in CANVAS PX at this Format — canvas px are display px, so
    # these are the numbers a reader's eye actually gets. compute() divides
    # by px_per_unit once and never mixes the two frames again.
    "rank_gap_px": 80.0,       # one step up the flow
    "lane_gap_px": 180.0,      # stream -> branch column
    "box_pad_px": (76.0, 24.0),   # clearance around a box's own label
    "pill_pad_px": (30.0, 24.0),  # the stream's two terminals
    "pill_min_px": (56.0, 37.1),  # matched to the branch-box height
    "add_radius_px": 22.0,
    "region_pad_px": 16.0,     # margin of a sublayer's grouping ground
    "region_opacity": 0.10,    # below the fill floor; the border carries it
    "clearance_px": 8.0,       # how close an edge may pass a foreign box
    "tap_frac": 0.45,          # read height, as a fraction of the rank gap
    "spine_width": 2.2,        # THE object: weight, not hue
    "branch_width": 1.0,
    "attn_stack": 2,           # ghost cards: the box abbreviates n heads
    "trunc_len_px": 44.0,      # the drawn stub of the continuing depth axis
    "expected_crossings": 0,
    "xlim": (-1.0, 1.0),       # pins the px-per-unit scale; see compute()
    "ylim": (-0.09, 2.22),
}


def compute(p):
    """Class B: no numerics. The rank layout is derived from the DAG, every
    spacing is a stated px distance, and assertions() checks the result."""
    scale = sch.px_per_unit(p["xlim"], FORMAT.display_width_px)
    size_pt = RISO.label_size_pt * FORMAT.ink_scale

    def u(px):                                  # canvas px -> math units
        return float(px) / scale

    lab = p["labels"]
    branch_keys = [k for tri in p["branches"] for k in tri[:2]]
    keys = p["spine"] + branch_keys

    # --- layout: rank = longest path, lane = which side of the stream -----
    layout = sch.RankLayout(axis="vertical", rank_gap=u(p["rank_gap_px"]),
                            lane_gap=u(p["lane_gap_px"]))
    ranks = sch.longest_path_ranks(keys, p["dag"])
    lanes = {k: 0 for k in p["spine"]}
    for ln, sub, _add, side in p["branches"]:
        lanes[ln] = lanes[sub] = side
    pos = sch.rank_positions(keys, layout=layout, ranks=ranks, lanes=lanes)

    # --- boxes sized from their own text, uniform across both columns -----
    box_min = (max(sch.auto_size(lab[k], scale, size_pt, p["box_pad_px"])[0]
                   for k in branch_keys), 0.0)
    nodes: dict[str, sch.Node] = {}
    for k in branch_keys:
        # the attention box abbreviates n heads: ghost cards, not a caption.
        # This figure's flow is upward, so the default down-right cards
        # would sit on the incoming edge and swallow its arrowhead; up-right
        # cards recede along the OUTGOING edge instead — the branch visibly
        # emerges from behind the stack.
        nodes[k] = sch.auto_node(k, pos[k], lab[k], scale=scale, size_pt=size_pt,
                                 pad_px=p["box_pad_px"], min_size=box_min,
                                 fill=RISO.background,
                                 stack=p["attn_stack"] if k == "attn" else 0,
                                 stack_dir=(1.0, 1.0))
    for k in ("in", "out"):
        w, h = sch.auto_size(lab[k], scale, size_pt, p["pill_pad_px"],
                             (u(p["pill_min_px"][0]), u(p["pill_min_px"][1])))
        # a pill: corner radius maxed, so the stream's ends read as terminals
        nodes[k] = sch.Node(k, pos[k], w, h, label=lab[k], corner=h / 2.0,
                            fill=RISO.background, n_arc=8)
    for k in ("add1", "add2"):
        nodes[k] = sch.circle_node(k, pos[k], u(p["add_radius_px"]), label="+",
                                   fill=RISO.background)

    dm = r"d_{\mathrm{model}}"
    edges: list[sch.Edge] = []

    # --- the spine: three segments of one line, the heaviest ink ----------
    spine_kw = dict(kind="excite", role=Role.CONTENT,
                    width_scale=p["spine_width"], head_scale=1.2)
    chain = p["spine"]
    for i, (a, b) in enumerate(zip(chain, chain[1:]), start=1):
        edges.append(sch.connect(nodes[a], nodes[b], key=f"spine{i}", **spine_kw))
    # the depth axis continues: the block REPEATS, and that used to ride
    # silently on the subscript. A truncated stub past x_{l+1} draws it —
    # hollow diamond, ellipsis: "goes on, deliberately stopped here"
    top = nodes["out"].port("top")
    edges.append(sch.edge(top, (top[0], top[1] + u(p["trunc_len_px"])),
                          truncated=True, px_per_unit=scale,
                          key="spine-continue", **spine_kw))
    # d_model on the first segment, in the empty gutter opposite branch 1
    e = edges[0]
    e.label, e.label_role = dm, Role.ANNOTATION
    e.label_anchor = (0.0, 0.5 * (pos["in"][1] + pos["add1"][1]))
    e.label_ha, e.label_va = ("left" if p["branches"][0][3] < 0 else "right"), "center"
    e.label_offset_px = (10.0 if p["branches"][0][3] < 0 else -10.0, 0.0)

    # --- the branches: read the stream, transform, write back -------------
    branch_kw = dict(role=Role.CONTENT, width_scale=p["branch_width"])
    taps: dict[str, tuple[float, float]] = {}
    for ln, sub, add, side in p["branches"]:
        tap_from = next(a for a, b in p["dag"] if b == ln)
        # READ: a junction on the stream, strictly between the node it taps
        # and the box it feeds, so the fork is visibly a branch off the line
        taps[ln] = (0.0, pos[tap_from][1] + p["tap_frac"] * u(p["rank_gap_px"]))
        edges.append(sch.edge(taps[ln], nodes[ln].port("bottom"), "map",
                              route="elbow", corner="hv",
                              key=f"read->{ln}", **branch_kw))
        edges.append(sch.connect(nodes[ln], nodes[sub], "map",
                                 key=f"{ln}->{sub}", **branch_kw))
        # WRITE: up to the rank of the +, then in from the side it lives on
        edges.append(sch.connect(
            (nodes[sub], "top"), (nodes[add], "left" if side < 0 else "right"),
            "map", route="elbow", corner=(pos[sub][0], pos[add][1]),
            key=f"{sub}->{add}", **branch_kw))

    # the second dimension label: the write is as wide as the stream, which
    # is the reason the + is defined at all
    w = edges[-1]
    w.label, w.label_role = dm, Role.ANNOTATION
    w.label_anchor = (0.5 * (w.anchors[1][0] + pos[p["branches"][-1][1]][0]),
                      pos[p["branches"][-1][2]][1])
    w.label_ha, w.label_va = "center", "bottom"
    w.label_offset_px = (0.0, -9.0)

    # --- the grouping ground: a sublayer is ONE thing ---------------------
    # The norm and the thing it feeds are not two stages in a pipeline; they
    # are one sublayer, and the pre-norm claim is about the whole of it. A
    # wash behind the pair says that with no ink the reader has to decode,
    # and it converts the claim from "no LayerNorm sits on the stream" into
    # the stronger "no part of either sublayer does" — which is what
    # assertions() now checks.
    regions = [sch.enclose(f"group:{sub}", [nodes[ln], nodes[sub]],
                           pad=u(p["region_pad_px"]), stroked=True,
                           opacity=p["region_opacity"])
               for ln, sub, _add, _side in p["branches"]]

    junctions = [Point(xy, role=Role.CONTENT, filled=True, radius_scale=1.15)
                 for xy in taps.values()]

    # the gutter opposite the upper branch is empty; name the object there
    annotations = [
        MathLabel(r"\text{residual stream}",
                  (-p["branches"][-1][3] * 0.32 * u(p["lane_gap_px"]),
                   pos[p["branches"][-1][1]][1]),
                  role=Role.ANNOTATION, ha="center", va="center",
                  angle_deg=90.0, pin=True),
    ]

    return {
        "params": p, "scale": scale, "size_pt": size_pt, "u": u,
        "ranks": ranks, "pos": pos, "nodes": nodes, "edges": edges,
        "taps": taps, "junctions": junctions, "annotations": annotations,
        "regions": regions, "xlim": p["xlim"], "ylim": p["ylim"],
    }


def build(g):
    s = Scene(xlim=g["xlim"], ylim=g["ylim"])
    s.add(*sch.assemble(list(g["nodes"].values()), g["edges"],
                        regions=g["regions"],
                        over=[*g["junctions"], *g["annotations"]]))
    return s


def assertions(g):
    from figlib.gates import Checks

    c = Checks()
    p, u = g["params"], g["u"]
    nodes, edges, pos, ranks = g["nodes"], g["edges"], g["pos"], g["ranks"]
    node_list = list(nodes.values())
    by_key = {e.key: e for e in edges}

    # --- the mechanism the figure claims -----------------------------------
    # 1. nothing runs backwards: the drawn y-order agrees with the ranking,
    #    so every arrow the reader follows points up the page
    for a, b in p["dag"]:
        c.check(pos[b][1] > pos[a][1] + 1e-9,
                f"{a}->{b} runs backwards: y {pos[a][1]:.3f} -> {pos[b][1]:.3f}")
    for _ln, sub, add, _side in p["branches"]:
        c.check(ranks[add] == ranks[sub] + 1,
                f"{add} does not sit one rank above {sub}")

    # 2. PRE-NORM: no LayerNorm is on the stream. That is the claim, and it
    #    is a statement about x, so it is checkable as one. The grouping
    #    ground makes it checkable of the WHOLE sublayer at once: if no part
    #    of either region reaches the spine lane, then nothing inside one
    #    can, whatever a later edit does to the boxes.
    x_spine = pos["in"][0]
    regions = g["regions"]
    for r in regions:
        rx0, _ry0, rx1, _ry1 = r.rect
        c.check(rx1 < x_spine or rx0 > x_spine,
                f"sublayer ground {r.key} straddles the residual-stream "
                f"lane at x = {x_spine} — that is post-norm")
    escaped = sch.region_containment_violations(regions, node_list)
    c.check(not escaped, "; ".join(escaped))
    tangled = sch.region_nesting_violations(regions)
    c.check(not tangled, "; ".join(tangled))
    c.check(len(regions) == len(p["branches"]),
            "a sublayer is missing its grouping ground")
    for ln, _sub, _add, _side in p["branches"]:
        c.check(abs(pos[ln][0] - x_spine) > nodes[ln].width / 2.0,
                f"{ln} overlaps the residual-stream lane — that is post-norm")
    c.check(all(abs(pos[k][0] - x_spine) < 1e-12 for k in p["spine"]),
            "a stream node is off the spine lane")
    sides = [np.sign(pos[sub][0] - x_spine) for _ln, sub, _a, _s in p["branches"]]
    c.check(len(set(sides)) == len(sides),
            "the branches are not on opposite sides — the stream is off-center")

    # 3. the spine really is ONE line: collinear, contiguous, heaviest
    spine = [by_key[f"spine{i}"] for i in (1, 2, 3)]
    for e in spine:
        c.check(np.allclose(e.pts[:, 0], x_spine, atol=1e-12),
                f"{e.key} is not vertical at x = {x_spine}")
        c.check(np.all(np.diff(e.pts[:, 1]) > 0), f"{e.key} does not run upward")
    branch = [e for e in edges if not e.key.startswith("spine")]
    c.check(min(e.width_scale for e in spine) > max(e.width_scale for e in branch),
            "the residual stream is not the heaviest ink in the figure")
    # the truncation stub is spine ink continuing the same line: it must
    # leave x_{l+1}'s top edge, stay on the lane, run upward, and its
    # ellipsis must land inside the declared limits (rank layout, not the
    # stub, decides where out's top is — so this CAN drift)
    depth = by_key["spine-continue"]
    c.check(np.allclose(depth.pts[:, 0], x_spine, atol=1e-12),
            f"the depth stub is not on the spine lane at x = {x_spine}")
    c.check(abs(depth.anchors[0][1] - nodes["out"].rect[3]) < 1e-9,
            "the depth stub does not leave x_{l+1}'s top edge")
    c.check(depth.anchors[1][1] > depth.anchors[0][1],
            "the depth stub does not continue upward")
    c.check(depth.width_scale == p["spine_width"],
            "the depth stub is not drawn at stream weight — it reads as "
            "an annotation, not the same line continuing")
    c.check(depth.anchors[1][1] + 4.0 * u(sch.TRUNC_DIAMOND_HALF_PX) < g["ylim"][1],
            "the truncation mark escapes the declared ylim")
    chain = p["spine"]
    for e, (a, b) in zip(spine, zip(chain, chain[1:])):
        c.check(abs(e.anchors[0][1] - nodes[a].rect[3]) < 1e-9,
                f"{e.key} does not leave {a}'s top edge")
        c.check(abs(e.anchors[1][1] - nodes[b].rect[1]) < 1e-9,
                f"{e.key} does not land on {b}'s bottom edge")

    # 4. every branch READS the stream below where it WRITES back, on a
    #    drawn segment, and each + takes exactly two inputs
    for ln, sub, add, side in p["branches"]:
        tap = g["taps"][ln]
        c.check(abs(tap[0] - x_spine) < 1e-12,
                f"the read for {sub} is not on the residual stream")
        c.check(any(e.pts[0, 1] <= tap[1] <= e.pts[-1, 1] for e in spine),
                f"the read for {sub} is not on a drawn spine segment")
        c.check(tap[1] < pos[add][1],
                f"{sub} writes below where it reads — the branch runs backwards")
        c.check(nodes[ln].rect[1] - tap[1] < u(p["rank_gap_px"]),
                f"the read for {sub} is not adjacent to the box it feeds")
        c.check(np.sign(by_key[f"{sub}->{add}"].anchors[1][0] - x_spine) == side,
                f"{sub} writes into the wrong side of {add}")
        incoming = [e for e in edges
                    if nodes[add].boundary_distance(e.anchors[1]) < 1e-9]
        c.check(len(incoming) == 2,
                f"{add} has {len(incoming)} inputs, not (stream + {sub})")

    # --- schematic checks ---------------------------------------------------
    ported = [e for e in edges
              if not e.key.startswith("read->") and e.key != "spine-continue"]
    off = sch.max_port_offset(ported, node_list)
    c.check(off < 1e-9, f"port attachment off the box boundary by {off:.3g}")
    for e in edges:
        if e.key.startswith("read->"):
            ln = e.key.split("->")[1]
            c.check(nodes[ln].boundary_distance(e.anchors[1]) < 1e-9,
                    f"{e.key} does not land on {ln}'s boundary")

    pierce = sch.clearance_violations(edges, node_list, pad=u(p["clearance_px"]))
    c.check(not pierce, "; ".join(pierce))

    n_cross = sch.crossing_count(edges)
    c.check(n_cross == p["expected_crossings"],
            f"{n_cross} edge crossings, declared {p['expected_crossings']}: "
            + "; ".join(f"{a}x{b}" for a, b, _ in sch.crossings(edges)))

    over = sch.label_overflow(node_list, g["scale"], g["size_pt"])
    c.check(not over, "; ".join(over))

    # every box inside the declared limits: the scene lims are stated, so a
    # box that grew past them would silently rescale the whole figure
    (xa, xb), (ya, yb) = g["xlim"], g["ylim"]
    for nd in node_list:
        x0, y0, x1, y1 = nd.rect
        c.check(xa <= x0 and x1 <= xb and ya <= y0 and y1 <= yb,
                f"node {nd.key} rect {tuple(round(v, 3) for v in nd.rect)} "
                f"escapes the declared limits")

    c.done()
