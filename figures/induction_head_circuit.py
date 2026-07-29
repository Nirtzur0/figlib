"""The induction-head circuit, drawn as rails and paths.

Design steps (architecture.md 0-9), recorded because for a Class B figure
the layout IS the content and there is no numerics to fall back on.

0. EARN IT. The prose version — "the induction head attends to the token
   that followed a previous occurrence of the current token, and copies
   it" — is a sentence people re-read three times. What it hides is that
   two different heads at two different layers compose: one writes a
   feature, the other keys on it. The inference the reader must make is
   *which position the attention lands on and why that position*, and on
   the page that becomes perceptual: the target of the QK arrow is a box
   whose printed content is the reason it was chosen.
1. CLAIM. Below.
2. REPRESENTATION. Token position on the x-axis (transformer-circuits'
   invariant: the graph reads like the prompt), one dashed rail per
   position for the residual stream, boxes only where something is
   written. Concrete instance throughout — real tokens A B C D A, not
   t_i — because a schematic of Greek letters installs no model
   (exposition.md, "ungrounded abstraction").
3. SIZE. WIDE. Five rails and two composed heads do not fit a column,
   and the annotation carries the mechanism.
4. TRAVERSAL. Enter bottom-left at the prompt; the low band of solid
   arcs is the previous-token head; the eye rises to the dashed arc
   sweeping right-to-left (the lookup) and then to the solid arc
   sweeping left-to-right (the copy). Backward-then-forward IS the
   circuit, and the two sweeps never cross — asserted, not hoped.
5. MECHANISM ANNOTATION. Readable off the figure: which position each
   head writes to, what is stored there (prev = A), what the query is
   matched against (q(A).k(prev=A)), and what gets copied (B).
6. READER EFFORT. Delegated: chaining "prev = A" to "the box sits on the
   rail of B" to "so B is what gets copied". Every premise is drawn; the
   one step is the reader's.
7. CHANNELS. Hue = head (the corpus's own discipline: two reserved hues
   for the two edge types). Indigo = previous-token head; brick =
   induction head. Kind = relation: `copy` (double chevron, content
   carried unchanged) vs `attend` (dashed, hollow head — a read, nothing
   written). Rails are construction ink; the ensemble of other
   previous-token writes is muted so ONE example path stays distinguished.
8. HONESTY. The figure draws a two-layer skeleton, not a transformer:
   MLPs, other heads, softmax weights on the non-selected positions, and
   the layer index of each head are all absent. It also draws attention
   as a single arrow where the real thing is a distribution — the muted
   ensemble at least admits that the previous-token head fires at every
   position, and "attn ~ 1" on the QK edge admits that the drawn arrow
   is the near-saturated case.
9. GATES. Numerical = the structural asserts below (the attend edge
   really lands on the position after the earlier A; the copied token
   really is the one at that position; ports exact; zero crossings; no
   edge pierces a box it is not attached to).
"""

import numpy as np

from figlib import schematic as sch
from figlib.format import WIDE
from figlib.scene import MathLabel, Scene, Vector
from figlib.style import Role
from figlib.theme import RISO

CLAIM = (
    "An induction head predicts the next token by attending to the position "
    "immediately AFTER an earlier occurrence of the current token — it can do "
    "that because a previous-token head has already written 'my predecessor "
    "was A' into that position's residual stream, which is exactly the key "
    "the induction head's query matches — and then copying whatever token "
    "sits there into the output."
)

THEME = RISO
FORMAT = WIDE

PARAMS = {
    "tokens": ["A", "B", "C", "D", "A"],
    "earlier": 0,          # the earlier occurrence of the current token
    "final": 4,            # the position that makes the prediction
    "dx": 2.4,             # rail spacing, math units
    "y_write": 1.15,       # where the previous-token head writes
    "y_query": 2.15,       # where the induction head's query is formed
    "y_out": 3.35,         # the prediction
    "rail_extent": (-0.30, 3.95),
    "expected_crossings": 0,
}


def _tt(t: str) -> str:
    return r"\mathtt{" + t + "}"


def compute(p):
    """Class B: no numerics — this IS the layout, stated coordinate by
    coordinate, and assertions() checks the statement."""
    toks = p["tokens"]
    n = len(toks)
    e, f = p["earlier"], p["final"]
    key_pos = e + 1                     # the position the induction head lands on
    xs = sch.columns(n, 0.0, p["dx"])
    y_w, y_q, y_o = p["y_write"], p["y_query"], p["y_out"]
    y_rail0 = p["rail_extent"][0]
    y_start = y_rail0 + 0.35            # arcs leave the rail just above its foot

    nodes = {
        # what the previous-token head wrote at key_pos — and therefore the
        # key the induction head matches. Indigo: written by the indigo head.
        # The box prints BOTH things this position's residual stream holds:
        # its own token (what the OV circuit will copy) and the feature the
        # previous-token head wrote (what the QK circuit will match). Printing
        # only the feature would leave "copies B" sourced from nowhere.
        "key": sch.Node("key", (xs[key_pos], y_w), 2.05, 0.56,
                        label=_tt(toks[key_pos]) + r",\;\text{prev}{=}" + _tt(toks[e]),
                        role=Role.ACCENT1, fill=RISO.background),
        # the induction head's query at the final position
        "query": sch.Node("query", (xs[f], y_q), 1.05, 0.56,
                          label="q(" + _tt(toks[f]) + ")",
                          role=Role.ACCENT2, fill=RISO.background),
        # the prediction
        "out": sch.Node("out", (xs[f], y_o), 1.42, 0.62,
                        label=r"\hat{y}{=}" + _tt(toks[key_pos]),
                        role=Role.ACCENT2, fill=RISO.background),
    }

    edges: list[sch.Edge] = []

    # --- previous-token head: copies token i into position i+1 -------------
    # Drawn as a population, so the reader cannot read it as a one-off: every
    # position gets the arc, ONE is accented (the one the circuit uses).
    for i in range(n - 1):
        src = (xs[i], y_start)
        accented = (i == e)
        dst = nodes["key"].port("left") if accented else (xs[i + 1], y_w)
        # the via biases the bulge low and late, so the arc leaves the rail
        # flat and turns up only under its target
        edges.append(sch.edge(
            src, dst, "copy", route="quad",
            via=(0.5 * (src[0] + dst[0]) + 0.20 * (dst[0] - src[0]), y_start + 0.22),
            key=f"prev{i}->{i + 1}", chevron_gap=0.36,
            role=Role.ACCENT1 if accented else Role.MUTED,
            # weight, not just hue, separates the actor from the ensemble
            width_scale=1.15 if accented else 0.7,
            head_scale=1.0 if accented else 0.78))

    # --- induction head: QK lookup, then OV copy ---------------------------
    # The lookup runs right-to-left (dashed, hollow: a read), the copy runs
    # left-to-right (solid, double chevron: content carried unchanged).
    edges.append(sch.edge(
        nodes["query"].port("left"), nodes["key"].port("right"), "attend",
        route="quad", via=(0.5 * (xs[key_pos] + xs[f]) + 0.05, y_q + 0.47),
        key="QK", role=Role.ACCENT2, label_halo=True,
        label=r"\text{QK}:\; q(" + _tt(toks[f]) + r")\cdot k(\text{prev}{=}"
              + _tt(toks[e]) + ")",
        label_anchor=(0.5 * (xs[key_pos] + xs[f]) + 0.45, y_q + 0.05),
        # +2px: the halo was grazing the QK edge's ink corridor by 1px
        label_offset_px=(0.0, 2.0),
        label_va="top"))

    edges.append(sch.edge(
        nodes["key"].port("top"), nodes["out"].port("left"), "copy",
        route="quad", via=(0.5 * (xs[key_pos] + xs[f]) - 0.4, y_o), key="OV",
        role=Role.ACCENT2, chevron_gap=0.36, label_halo=True,
        label=r"\text{OV}:\;\text{copies}\;" + _tt(toks[key_pos]),
        label_anchor=(0.5 * (xs[key_pos] + xs[f]) - 0.10, y_o + 0.35),
        label_va="bottom"))

    rail_items = sch.rails(
        xs, p["rail_extent"], labels=[_tt(t) for t in toks],
        label_role=Role.CONTENT, label_offset_px=(0.0, 12.0))

    labels = [
        # the pattern the circuit implements, in the same token notation
        MathLabel(_tt(toks[e]) + r"\," + _tt(toks[key_pos]) + r"\;\ldots\;"
                  + _tt(toks[f]) + r"\;\Rightarrow\;" + _tt(toks[key_pos]),
                  (-0.35, y_o + 0.27), ha="left", va="bottom", role=Role.CONTENT,
                  halo=True),
        MathLabel(r"\text{residual stream}", (xs[0] + 0.18, 2.18), ha="left",
                  va="bottom", role=Role.ANNOTATION, halo=True),
        # the vertical axis is computation depth; an unlabelled axis would be
        # an unencoded position the reader reads as a claim anyway
        MathLabel(r"\text{depth}", (-0.62, 1.25), ha="left", va="bottom",
                  role=Role.ANNOTATION, angle_deg=90.0, offset_px=(-8.0, 0.0)),
        MathLabel(r"\text{previous-token head}", (0.5 * (xs[1] + xs[3]) + 0.05, 0.58),
                  ha="center", va="bottom", role=Role.ACCENT1, halo=True),
        MathLabel(r"\text{induction head}", (xs[1] + 0.55, y_o - 0.37),
                  ha="center", va="bottom", role=Role.ACCENT2, halo=True),
        # honesty: the drawn arrow is the near-saturated case of what is
        # really a distribution over every earlier position
        MathLabel(r"\text{attn} \approx 1",
                  (0.5 * (xs[key_pos] + xs[f]) + 0.45, y_q - 0.32),
                  ha="center", va="top", role=Role.ACCENT2, halo=True),
    ]

    return {
        "tokens": toks, "xs": xs, "nodes": nodes, "edges": edges,
        "rail_items": rail_items, "labels": labels,
        "earlier": e, "final": f, "key_pos": key_pos,
        "y_write": y_w, "y_query": y_q, "y_out": y_o,
        "expected_crossings": p["expected_crossings"],
        "xlim": (-0.9, 10.6), "ylim": (-1.05, 4.10),
    }


def build(g):
    s = Scene(xlim=g["xlim"], ylim=g["ylim"])
    s.add(Vector((-0.62, 0.15), (-0.62, 3.30), role=Role.ANNOTATION,
                 width_scale=0.75))
    s.add(*g["rail_items"])
    # muted ensemble first, accented paths over it, boxes last so their
    # paper fill masks the rails running behind them
    for e in g["edges"]:
        if e.role is Role.MUTED:
            s.add(*e.items())
    for e in g["edges"]:
        if e.role is not Role.MUTED:
            s.add(*e.items())
    for nd in g["nodes"].values():
        s.add(*nd.items())
    s.add(*g["labels"])
    return s


def assertions(g):
    from figlib.gates import Checks

    c = Checks()
    toks, xs = g["tokens"], g["xs"]
    e, f, k = g["earlier"], g["final"], g["key_pos"]
    nodes = list(g["nodes"].values())
    edges = g["edges"]
    by_key = {ed.key: ed for ed in edges}

    # --- the mechanism the figure claims -----------------------------------
    c.check(toks[e] == toks[f],
            f"the earlier token {toks[e]!r} is not the current token {toks[f]!r}")
    c.check(k == e + 1, f"the key position {k} is not the one AFTER the earlier {e}")
    c.check(k < f, "the attended position must precede the predicting position")
    c.check(g["nodes"]["out"].label.endswith(_tt(toks[k]) ),
            "the predicted token is not the token at the attended position")

    qk, ov = by_key["QK"], by_key["OV"]
    # the QK edge really targets the box on the rail of position k, and the
    # OV edge really leaves that same box
    c.check(abs(qk.anchors[1][0] - g["nodes"]["key"].rect[2]) < 1e-9
            and abs(qk.anchors[1][1] - g["y_write"]) < 1e-9,
            "QK does not land on the 'prev' box's right port")
    c.check(abs(ov.anchors[0][0] - xs[k]) < 1e-9,
            f"OV does not leave the rail of position {k}")
    c.check(abs(ov.anchors[1][0] - g["nodes"]["out"].rect[0]) < 1e-9,
            "OV does not land on the output box")
    # direction: the lookup runs backward, the copy runs forward
    c.check(np.all(np.diff(qk.pts[:, 0]) < 0), "the QK edge does not run backward")
    c.check(np.all(np.diff(ov.pts[:, 0]) > 0), "the OV edge does not run forward")

    # every previous-token arc really goes from rail i to rail i+1
    for i in range(len(toks) - 1):
        ed = by_key[f"prev{i}->{i + 1}"]
        c.check(abs(ed.anchors[0][0] - xs[i]) < 1e-9,
                f"prev-token arc {i} does not start on rail {i}")
        target_x = (g["nodes"]["key"].rect[0] if i == e else xs[i + 1])
        c.check(abs(ed.anchors[1][0] - target_x) < 1e-9,
                f"prev-token arc {i} does not land at position {i + 1}")

    # --- schematic checks ---------------------------------------------------
    # QK and OV run box-to-box, so both their anchors must sit exactly on a
    # boundary; the previous-token arcs start on a bare rail, so only their
    # accented landing is a port.
    off = max(sch.max_port_offset([qk, ov], nodes),
              g["nodes"]["key"].boundary_distance(
                  by_key[f"prev{e}->{e + 1}"].anchors[1]))
    c.check(off < 1e-9, f"port attachment off the box boundary by {off:.3g}")

    pierce = sch.clearance_violations(edges, nodes, pad=0.06)
    c.check(not pierce, "; ".join(pierce))

    n_cross = sch.crossing_count(edges)
    c.check(n_cross == g["expected_crossings"],
            f"{n_cross} edge crossings, declared {g['expected_crossings']}: "
            + "; ".join(f"{a}x{b}" for a, b, _ in sch.crossings(edges)))

    # labels fit the boxes they were declared with (sizes are stated, not
    # derived — this is what keeps the declaration honest)
    scale = sch.px_per_unit(g["xlim"], FORMAT.display_width_px)
    over = sch.label_overflow(nodes, scale, RISO.label_size_pt * FORMAT.ink_scale)
    c.check(not over, "; ".join(over))

    c.done()
