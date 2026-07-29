# Honesty Marks & Composition Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the remaining capability gaps identified by the corpus study: honesty devices drawn as marks (elision, truncation, unknown-mechanism), the labelled operator junction, the typographic register channel (mono/sans), the hue-as-referential-noun gate, word-scale insets (scene-in-scene embedding), and the annotated-derivation builder — each landing with a benchmark figure that forces it.

**Architecture:** Every capability enters as a producer of scene items (never a new renderer). Schematic marks extend `schematic.py`'s existing `Node`/`Edge` item emission. The register channel is a LaTeX-wrapping concern inside `typeset`-adjacent code, so metrics and gates come free. The inset is a pure affine mapping of one Scene's items into another's math coords. The derivation row is a builder over typeset metrics. The hue gate extends the existing color gate over `key`ed items.

**Tech Stack:** Python 3, numpy, ziamath, cairosvg; pytest with `tests/svgkit.py` for SVG assertions.

## Global Constraints

Copied from `CLAUDE.md` / `docs/architecture.md` — every task's requirements include these:

- **Content code never names a color, font, or stroke width.** It names meanings: `Role.*`, `theme.ramp(t)`, `theme.categorical(i)`. A hex literal in a figure program is a defect.
- **New capability enters as a producer of scene items, never as a new renderer.**
- **The `*_ink()` rule:** a new item whose ink is derived (not literal points) ships ONE canvas-px resolver in `render.py` returning geometry + any derived MathLabels; `gates.py` imports it. A gate that re-derives geometry is a bug.
- **Ink math is single-sourced in render.py.** A producer module that needs it imports lazily inside the function. Never a module-level back-import.
- **Canvas units are display CSS pixels.** Never shrink type to make something fit.
- **Assert what could be wrong, never what is true by construction.**
- **Tests assert SVG via `tests/svgkit.py`** (`tag()`, `find_by()`, `path_cmd_counts()`) — never raw `e.tag` (ElementTree namespaces every tag).
- **Rendering/gating a figure requires the make targets** (`make check F=figures/x.py`, `make test`, `make regress`) — bare `figcheck` and bare pytest without `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib` fail on cairo.
- **`figures/out/` diffs are reviewed, then `make update`.** Any diff there is a real render change.
- **A new figure is done** when gates pass, `make regress` is clean, and a readback record exists (`figures/out/<name>.readback.md`).
- **TDD:** failing test first, then implementation, then commit. Numerical-geometry helpers get numeric tests; item producers get item-level tests plus (where they render) svgkit tests.
- Implementers MUST read `docs/skill.md` before writing or editing any figure program, and the "Module-authoring rules" section of `docs/primitive-gaps.md` before touching `src/figlib/`.

## Scope decisions (made deliberately; do not re-open)

- **Declared exaggeration** is a convention, not an API: an ANNOTATION-role label stating the distortion and its vanishing order, placed by the figure program. It lands in the docs task (Task 11) as a named device with the benchmark figures as exemplars.
- **Small multiples** get NO new API. `Figure` (grid) + `Correspondence` already carry the semantics; the deliverable is the benchmark exemplar (Task 10) that the skill can imitate. Adding a helper would be a template, which `docs/primitive-gaps.md` §"Design doctrine" forbids.
- **Joint label placement** stays out of scope — `docs/primitive-gaps.md` records that fixing it means a constraint solver, which the doctrine forbids; it needs a human decision, not code.

---

### Task 1: Schematic honesty marks — elision stack, unknown-mechanism node, declared truncation

**Files:**
- Modify: `src/figlib/schematic.py` (Node fields + items; Edge truncation)
- Test: `tests/test_schematic.py` (append)

**Interfaces:**
- Produces: `Node(..., stack: int = 0, dash: str | None = None)`; `Edge(..., truncated: bool = False)`; module constants `STACK_OFFSET`, `TRUNC_DIAMOND_HALF`.
- Consumed by Task 7 (benchmark figure).

Semantics (from `docs/primitive-gaps.md` §"Still missing"):
- `stack=n` (n ≥ 1): n ghost cards behind the node — the stacked-card shadow meaning "this node abbreviates many". Each ghost is the node's outline offset by `(k * STACK_OFFSET, -k * STACK_OFFSET)` in math coords (up-right, so ghosts peek out top-right), drawn BEFORE the node itself, paper-filled if the node has `fill`, else outline-only, at `Role.MUTED`. `STACK_OFFSET` is a fraction of min(width, height): `0.10 * min(w, h)`.
- `dash` on Node: passes through to the outline `Curve.dash` (and to the FilledCurve's outline when filled — check how FilledCurve outlines render; if FilledCurve has no dash channel, emit the outline as a separate Curve with `dash` on top of an outline-free fill). The unknown-mechanism node is then written by a figure as `Node(key, center, w, h, label=r"?\,?\,?", dash="dashed")` — add a module-level convenience `unknown_node(key, center, width, height, **kw) -> Node` that fills in exactly those two settings.
- `Edge(truncated=True)`: the edge declares "this continues and I am choosing to stop." Replace the terminal head(s) with a small hollow diamond centered at the tip, axis-aligned to the terminal tangent, half-diagonal `TRUNC_DIAMOND_HALF = 5.0` (math-unit scale: use the same units `bar_half` uses — read `BAR_HALF`'s units first and match), followed by a `\cdots` MathLabel just past the tip along the tangent (offset by ~2.5 × the half-diagonal), ANNOTATION role. `head_fractions()` returns `()` when truncated.

- [ ] **Step 1: Write failing tests**

```python
def test_node_stack_emits_ghost_cards_behind():
    n = Node("mha", (0.0, 0.0), 4.0, 2.0, label="MHA", stack=2)
    items = n.items()
    curves = [it for it in items if isinstance(it, (Curve, FilledCurve))]
    # 2 ghosts + 1 body outline, ghosts first (drawn behind)
    assert len(curves) == 3
    assert curves[0].role == Role.MUTED and curves[1].role == Role.MUTED
    # ghosts offset up-right by k * 0.10 * min(w, h)
    off = 0.10 * 2.0
    assert np.allclose(curves[1].pts, curves[2].pts + [off, -off]) or \
           np.allclose(curves[0].pts, curves[2].pts + [2 * off, -2 * off])

def test_unknown_node_is_dashed_with_mystery_label():
    n = unknown_node("mystery", (0.0, 0.0), 3.0, 1.5)
    items = n.items()
    labels = [it for it in items if isinstance(it, MathLabel)]
    assert labels and "?" in labels[0].latex
    curves = [it for it in items if isinstance(it, Curve)]
    assert any(c.dash == "dashed" for c in curves)

def test_truncated_edge_diamond_and_ellipsis():
    e = edge((0.0, 0.0), (10.0, 0.0), "map", truncated=True)
    items = e.items()
    main = items[0]
    assert main.arrows == ()          # no head — the diamond replaces it
    labels = [it for it in items if isinstance(it, MathLabel)]
    assert any(r"\cdots" in l.latex for l in labels)
    # the diamond: a closed 4-point curve near the tip
    diamonds = [it for it in items if isinstance(it, Curve) and it.closed
                and len(it.pts) in (4, 5)]
    assert diamonds and np.allclose(np.mean(diamonds[0].pts[:4], axis=0),
                                    [10.0, 0.0], atol=1.0)
```

- [ ] **Step 2: Run tests, verify they fail** (`make test` or targeted `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib uv run pytest tests/test_schematic.py -k "stack or unknown or truncated" -v`)
- [ ] **Step 3: Implement** — Node.stack/dash + `unknown_node`, Edge.truncated per the semantics above. Diamond is literal points (a producer emission, not derived ink at render time), so NO render.py resolver is needed.
- [ ] **Step 4: Tests pass; run the full suite** (`make test`)
- [ ] **Step 5: Commit** `feat(schematic): elision stack, unknown-mechanism node, declared truncation`

---

### Task 2: Labelled operator junction

**Files:**
- Modify: `src/figlib/schematic.py`
- Test: `tests/test_schematic.py` (append)

**Interfaces:**
- Produces: `Junction(key, center, glyph, radius=..., args=((port_spec_angle_deg, latex), ...), role=..., fill=...)` with `.items() -> list[Item]` and `.port(angle_deg) -> XY`.
- Consumed by Task 7.

Semantics (from `docs/primitive-gaps.md`): a named binary (or n-ary) operator sitting ON an edge junction — the corpus's K/Q glyph — with its argument roles annotated. It generalizes the circled `+`, which is role-blind.

Design:
- `Junction` is a frozen dataclass, NOT a `_Box` (it is circular). Fields: `key: str`, `center: XY`, `glyph: str` (LaTeX, e.g. `r"\odot"` or `r"\mathrm{QK}"`), `radius: float`, `args: tuple[tuple[float, str], ...] = ()` — each `(angle_deg, latex)` places a small ANNOTATION label just OUTSIDE the circle at that angle (angle 0 = +x, CCW), `role: Role = Role.CONTENT`, `fill: str | None = None` (paper color, same convention as `Node.fill`), `label_size_pt: float | None = None`, `arg_size_pt: float | None = None` (default: small — pass ~8.5–9 pt in figures, but the field default stays None → style default).
- `.port(angle_deg) -> XY`: point on the circle at that angle — edges attach there.
- `.items()`: circle as closed Curve (or paper FilledCurve + outline when `fill`), glyph MathLabel centered (`ha="center", va="center"`), arg labels at `center + 1.45 * radius * (cos, sin)`, each with ha/va chosen by octant (right half → `ha="left"`, left half → `ha="right"`, near-vertical → `ha="center"`; va analogous) so text grows away from the circle.
- Whether `clearance_violations` should treat a Junction as an obstruction: YES — mirror however that check collects `Node` outlines; register the junction's circle (its bounding box is fine if the check is box-based). Read the check before wiring; do not fork its logic.

- [ ] **Step 1: Failing tests** — circle radius honored; glyph label centered; arg label at the stated angle with outward ha/va; `.port(0.0) == center + (radius, 0)`.

```python
def test_junction_glyph_and_arg_roles():
    j = Junction("qk", (2.0, 3.0), r"\mathrm{QK}", radius=0.6,
                 args=((90.0, "q"), (180.0, "k")))
    items = j.items()
    labels = [it for it in items if isinstance(it, MathLabel)]
    glyph = next(l for l in labels if l.latex == r"\mathrm{QK}")
    assert glyph.ha == "center" and glyph.va == "center"
    q = next(l for l in labels if l.latex == "q")
    assert np.allclose(q.anchor, (2.0, 3.0 + 1.45 * 0.6), atol=1e-9)
    assert j.port(0.0) == (2.6, 3.0)
```

- [ ] **Step 2: Verify failure**
- [ ] **Step 3: Implement**
- [ ] **Step 4: Full `make test` passes**
- [ ] **Step 5: Commit** `feat(schematic): labelled operator junction`

---

### Task 3: Typographic register channel (mono / sans)

**Files:**
- Modify: `src/figlib/scene.py` (MathLabel field), `src/figlib/typeset.py` (register wrapper), `src/figlib/render.py` + `src/figlib/gates.py` ONLY at the call sites where `MathLabel.latex` reaches `render_math`/`draw_math` (apply the wrapper so metrics and drawing agree)
- Modify: `src/figlib/schematic.py` (`Node.label_register`, `Edge.label_register` pass-through)
- Test: `tests/test_scene.py` or a new `tests/test_register.py`

**Interfaces:**
- Produces: `MathLabel(..., register: str | None = None)` with values `None | "mono" | "sans"`; `typeset.apply_register(latex: str, register: str | None) -> str`.
- Consumed by Task 7 (mono token strip).

Semantics (from `docs/primitive-gaps.md` §containment-and-elision): typeface encodes epistemic status — monospace = literal model input / data, sans = human interpretation, default math serif = mathematics. This is a semantic channel like Role, so it lives on the label, and themes do not see it (it is not appearance-varying).

Design — one typeset path, zero new metrics code:
```python
_REGISTERS = {"mono": r"\mathtt", "sans": r"\mathsf"}

def apply_register(latex: str, register: str | None) -> str:
    if register is None:
        return latex
    try:
        cmd = _REGISTERS[register]
    except KeyError:
        raise ValueError(f"unknown register {register!r}; have {sorted(_REGISTERS)}")
    return f"{cmd}{{{latex}}}"
```
Wrap at EVERY point where a MathLabel's latex is measured or drawn (render and gates share the resolver path — find the single choke point(s); if measurement happens in `gates.py` via `render_math`, both must wrap identically or bboxes drift from ink). Grep for `render_math(` and `draw_math(` call sites that take a `MathLabel` and thread `apply_register(l.latex, l.register)` through. Verify ziamath renders `\mathtt{...}`/`\mathsf{...}` — write one smoke test that calls `render_math(apply_register("Q", "mono"))` and asserts nonzero metrics that differ from the serif width.

- [ ] **Step 1: Failing tests** — `apply_register` wrapping + error case; a MathLabel with `register="mono"` renders (svgkit: label present) and its measured width differs from the unregistered width; schematic pass-throughs set the field.
- [ ] **Step 2: Verify failure**
- [ ] **Step 3: Implement**
- [ ] **Step 4: `make test` + `make regress`** (regress must be CLEAN — no existing figure sets register)
- [ ] **Step 5: Commit** `feat(typeset): mono/sans register channel on MathLabel`

---

### Task 4: Hue-as-referential-noun gate

**Files:**
- Modify: `src/figlib/gates.py` (or `src/figlib/correspond.py` if the cross-panel walk lives there — read both first and put the check where keyed items are already enumerated)
- Test: `tests/test_correspondence.py` (append) or `tests/test_gates.py`

**Interfaces:**
- Produces: a gate check `hue_binding_violations(...)` reported alongside the existing correspondence residual; two violation kinds: `hue-split` and `hue-collision`.

Semantics (from `docs/corpus-study.md` §6 mechanisms, "Color as referential noun"): one hue per named object, declared once, enforced everywhere. Two failure modes:
1. **hue-split**: items sharing a `key` (within one Scene or across a Figure's panels) resolve to DIFFERENT colors — the same object drawn in two hues. First check what `correspond.py`'s fingerprint already covers: if color is already part of the identity fingerprint for cross-panel keys, the new work is only the within-scene check and the collision check below; do not duplicate an existing residual.
2. **hue-collision**: items with DIFFERENT keys carry the SAME `Hue` (the tagged str subclass from `theme.categorical` — see `docs/architecture.md` §Themes) — two named objects sharing a correspondence hue, which breaks referential reading. Only `Hue`-tagged colors from the categorical channel participate (a bare `#rrggbb` is not claiming identity; ramp/shade hues are ordered quantity, not nouns). Only KEYED items participate — an unkeyed decorative use of `categorical` is not a claim.

Resolution rule for "the item's color": the same precedence render uses (item `.color` override, else the Role's ink). Compare resolved color strings, case-insensitive.

- [ ] **Step 1: Failing tests** — same key two colors → one `hue-split`; two keys same categorical hue → one `hue-collision`; unkeyed items and non-Hue colors never fire; clean figure → no violations.
- [ ] **Step 2: Verify failure**
- [ ] **Step 3: Implement; wire into the gate runner's report** (same registration pattern as `region_containment_violations` — find it and mirror)
- [ ] **Step 4: `make test` + `make regress`** (existing corpus must stay clean; if a real figure fires the gate, STOP and report — that is either a latent bug in the figure or an over-eager check, and the reviewer decides)
- [ ] **Step 5: Commit** `feat(gates): hue-binding gate — hue is a referential noun`

---

### Task 5: Word-scale inset — scene-in-scene embedding

**Files:**
- Create: `src/figlib/inset.py`
- Test: `tests/test_inset.py`

**Interfaces:**
- Produces: `embed(small: Scene, *, at: XY, width: float, frame: bool = True, leader_to: XY | None = None, key: str | None = None) -> list[Item]` — items in HOST math coords, ready for `host.add(*embed(...))`.
- Consumed by Task 9.

Semantics (from `docs/corpus-study.md`, "word-scale insets"): a tiny concrete graphic embedded at the exact point in a larger diagram where the object lives. This is a PRODUCER: it affinely maps the small scene's items into host coordinates — no new renderer, no nested transforms, gates apply to the result for free.

Design:
- The small scene must have explicit `xlim`/`ylim` (raise `ValueError` otherwise — the mapping needs a source rect).
- Mapping: source rect `(xlim, ylim)` → dest rect centered at `at` with width `width` and height `width * (Δy/Δx)` (equal aspect; if `small.height_px` is set, raise — px-height scenes have no math aspect and are out of scope for v1).
- Transform every coordinate field: `Curve.pts`, `FilledCurve.pts` + `holes`, `Vector.tail/tip`, `Point.xy`, `MathLabel.anchor`, `Brace.p1/p2` (+ scale `depth` if set), `Callout.anchor/target`, `AngleMark.center` (+ scale `radius`), `RightAngleMark.corner` (+ scale `size`), `RasterField.extent`, `Gradient.p0/p1` inside FilledCurve. Scale `Brace.depth`, `AngleMark.radius`, `RightAngleMark.size` by the linear scale factor. **Do NOT scale** `size_pt`, `offset_px`, `width_scale`, `radius_scale`, `arrow_scale` — type and ink stay at reading size by the canvas-px invariant; a too-dense inset is caught by the mechanical gate, and the fix is a bigger inset or less ink, never smaller type.
- Use `dataclasses.replace` on copies; never mutate the small scene's items.
- `frame=True`: a rounded-rect closed Curve at the dest rect bounds, `Role.FRAME`.
- `leader_to`: a straight `Role.ANNOTATION` Curve from the nearest point of the dest rect boundary to `leader_to`, with a small filled arrowhead — reuse the Callout leader's approach: if Callout's leader ink is derived in `render.py`, this one should be literal points instead (compute the arrowhead triangle in math coords via the same geometry helpers `figure.py` uses lazily — `from .render import _arrowhead` inside the function; note `_arrowhead` works in canvas px in figure.py's usage — if px/math mixing is unsound here, emit the leader as a `Vector` instead, which already owns its head). Choose ONE and say why in the module docstring.
- `key` forwarded onto every produced item that has a `key` field (the inset as a whole can then participate in correspondence).
- Clipping: v1 does NOT clip inset content to its frame; content leaking past the frame is caught by eye and by the frame being FRAME-role ink. Record this in the docstring as a known limit.

- [ ] **Step 1: Failing tests**

```python
def test_embed_maps_corners_exactly():
    small = Scene(xlim=(0, 2), ylim=(0, 1))
    small.add(Curve(np.array([[0, 0], [2, 1]]) * 1.0))
    out = embed(small, at=(10.0, 5.0), width=1.0, frame=False)
    c = next(it for it in out if isinstance(it, Curve))
    # dest rect: 1.0 wide, 0.5 tall, centered at (10, 5)
    assert np.allclose(c.pts[0], [9.5, 4.75]) and np.allclose(c.pts[1], [10.5, 5.25])

def test_embed_preserves_type_size_and_scales_geometry():
    small = Scene(xlim=(0, 1), ylim=(0, 1))
    small.add(MathLabel("x", (0.5, 0.5), size_pt=9.0),
              AngleMark((0.5, 0.5), (1, 0), (0, 1), radius=0.2))
    out = embed(small, at=(0.0, 0.0), width=0.5, frame=False)
    lab = next(it for it in out if isinstance(it, MathLabel))
    assert lab.size_pt == 9.0                      # type never scales
    am = next(it for it in out if isinstance(it, AngleMark))
    assert np.isclose(am.radius, 0.1)              # geometry does

def test_embed_requires_lims_and_equal_aspect():
    with pytest.raises(ValueError):
        embed(Scene(), at=(0, 0), width=1.0)
    s = Scene(xlim=(0, 1), ylim=(0, 1), height_px=120.0)
    with pytest.raises(ValueError):
        embed(s, at=(0, 0), width=1.0)

def test_embed_does_not_mutate_source():
    small = Scene(xlim=(0, 1), ylim=(0, 1))
    pts = np.array([[0.0, 0.0], [1.0, 1.0]])
    small.add(Curve(pts))
    embed(small, at=(5.0, 5.0), width=2.0)
    assert np.allclose(small.items[0].pts, pts)
```

- [ ] **Step 2: Verify failure**
- [ ] **Step 3: Implement** (single dispatch table over item types; unknown item type → `TypeError` naming it, so the next primitive added to scene.py fails loudly here instead of silently passing through unmapped)
- [ ] **Step 4: `make test` passes**
- [ ] **Step 5: Commit** `feat(inset): word-scale scene embedding as an item producer`

---

### Task 6: Annotated derivation builder

**Files:**
- Create: `src/figlib/derivation.py`
- Test: `tests/test_derivation.py`

**Interfaces:**
- Produces: `Term(latex: str, gloss: str | None = None, role: Role = Role.CONTENT, key: str | None = None)`; `derivation_row(terms: Sequence[Term], *, lhs: str | None = None, op: str = "+", y: float = 0.0, size_pt: float | None = None, gloss_size_pt: float | None = None, px_per_unit: float, gap_pt: float = 14.0) -> tuple[list[Item], list[XY]]` — the items and the per-term center anchors (for figure programs that want to add more annotation).
- Consumed by Task 8.

Semantics (from `docs/corpus-study.md` §"annotated derivation"): the equation IS the figure — each term of an expansion carries a prose gloss anchored beneath it. Typography does the layout.

Design:
- Layout happens in MATH units but is METRIC in px: caller states `px_per_unit` (figures compute it from their FORMAT width and xlim — the same convention `plots.px_units` serves; read it and reference it in the docstring). Widths come from `typeset.render_math(apply_register(...)).width_px` at the resolved size; terms are placed left-to-right: `lhs =` (if given), then term, op, term, … with `gap_pt` (converted to math units) around operators. All MathLabels are `pin=True` (position IS meaning here) with `ha="center", va="center"`, centered at their measured half-widths.
- Glosses: beneath each glossed term at a fixed drop (`1.8 ×` the term's height in math units below `y`), `Role.ANNOTATION`, `register="sans"` (Task 3's channel — the gloss is interpretation, the math is math), centered on the term, with a short vertical tick Curve (ANNOTATION, ~0.6 of the drop) from below the term to above the gloss. Multi-glossed rows with long glosses will collide; that is the mechanical gate's job to report, and the figure program's job to shorten prose — the builder does NOT solve packing (doctrine: no constraint solvers).
- Return anchors so programs can brace groups of terms with the existing `Brace`.

- [ ] **Step 1: Failing tests** — terms placed strictly left-to-right, non-overlapping at the metric widths (assert `x_{i+1} - x_i >= (w_i + w_{i+1})/2` in math units); ops midway between neighbors; gloss anchored below its term center with `register == "sans"`; every produced MathLabel pinned; anchors list matches term count.
- [ ] **Step 2: Verify failure** (needs cairo env: run via `make test`)
- [ ] **Step 3: Implement**
- [ ] **Step 4: `make test` passes**
- [ ] **Step 5: Commit** `feat(derivation): annotated-derivation row builder`

---

### Task 7: Benchmark — transformer block with honesty marks, junction, and mono register

**Files:**
- Modify: `figures/schematic_transformer_block.py`
- Baselines: `figures/out/schematic_transformer_block.{svg,png}` (+ readback record refresh)

**Interfaces:** Consumes Tasks 1–3 exactly as specified there.

Read `docs/skill.md` FIRST, then the current figure program end-to-end. Apply, per the corpus lesson ("where our figures admit an elision in the docstring, theirs admit it on the page"):
- Multi-Head Attention node → `stack=2` (it abbreviates n heads; delete the docstring apology and note the mark in the module docstring instead).
- The block-repetition axis (× N layers, however the figure currently states it) → a `truncated=True` edge continuing the depth axis, replacing any `\times N`-style caption if one exists (or adding the truncation where repetition is currently implied silently).
- If the figure contains an attention-score computation point, mark it with a `Junction` (glyph `r"\mathrm{QK}"`, args `q`/`k`) — only if it fits the figure's existing claim; do NOT force it. If it does not fit, note that in the report and leave the junction to the readback of `induction_head_circuit` in a follow-up (do not modify that figure in this task).
- If the figure carries a literal token/prompt strip (or should, per the corpus's "anchored to a literal monospace strip"), set `register="mono"` on it.
- Update the figure's assertions for anything now assertable (e.g. the truncation edge's existence is by construction — do NOT assert it; assert only data-dependent geometry).

- [ ] **Step 1: `make check F=figures/schematic_transformer_block.py` passes** after edits (iterate on gate diagnostics — they contain the fix)
- [ ] **Step 2: `make regress`** — inspect the `figures/out/` diff, confirm every change is intended, then `make update`
- [ ] **Step 3: Readback record** — `figcheck --readback-prompt` via the raw uv command, dispatch a cold read, `readback.record()` → `figures/out/schematic_transformer_block.readback.md`
- [ ] **Step 4: Commit** `figures: transformer block draws its elisions — stack, truncation, junction, mono strip`

---

### Task 8: Benchmark — QK score expansion as an annotated derivation

**Files:**
- Create: `figures/qk_score_derivation.py`
- Baselines: `figures/out/qk_score_derivation.{svg,png}` + readback record

**Interfaces:** Consumes Task 6 (`derivation_row`) and Task 3 (`register="sans"` glosses arrive via the builder).

The claim (write it as the program's CLAIM constant, per `docs/skill.md`): the QK attention score decomposes exhaustively into interaction terms, each with a mechanistic meaning. Content: expand `s = (x_q + ε_q)ᵀ W (x_k + ε_k)` into four terms (signal–signal, signal–error, error–signal, error–error) with a gloss under each naming what it does; brace the two cross terms with the existing `Brace` labeled as the interaction pair. FORMAT = WIDE (a derivation row is wide and short). No decoration: the equation and its glosses are the entire figure.

- [ ] **Step 1: Write the program** (design steps 0–9 from `docs/architecture.md` in the module docstring, as `docs/skill.md` prescribes)
- [ ] **Step 2: `make check F=figures/qk_score_derivation.py --report` passes**
- [ ] **Step 3: `make regress` clean for the rest of the corpus; `make update` adds the new baselines**
- [ ] **Step 4: Readback record** written
- [ ] **Step 5: Commit** `figures: QK score expansion — the annotated derivation benchmark`

---

### Task 9: Benchmark — saddle-node behavior map with pinned phase-line insets

**Files:**
- Create: `figures/saddle_node_behavior_map.py` (do NOT modify `figures/strogatz_saddle_node.py` — it is a committed benchmark of a different device)
- Baselines + readback record as usual

**Interfaces:** Consumes Task 5 (`embed`) and existing `plots.phase_line` / `plots.flow_intervals`.

The claim: the bifurcation diagram of ẋ = r + x² is a *map of behaviors* — at each r the system IS a phase line, and the diagram compiles them (Victor's behavior map; `docs/corpus-study.md` runner-up device). Content: the r–x bifurcation diagram (solid stable / dashed unstable branches, existing conventions from the strogatz figure — read it for the house idiom) with two or three word-scale phase-line insets embedded at chosen r values (r < 0, r = 0, r > 0), each a tiny Scene built from `plots.phase_line`, framed, with a leader to its r on the axis. Key the branch curves and reuse their hue in the insets' fixed points (Task 4's gate then holds the binding — this benchmark exercises it for real).

- [ ] **Step 1: Write the program** (design steps 0–9 in the docstring)
- [ ] **Step 2: `make check` passes; iterate on diagnostics**
- [ ] **Step 3: `make regress` clean; `make update` for the new baselines**
- [ ] **Step 4: Readback record** written
- [ ] **Step 5: Commit** `figures: saddle-node behavior map — word-scale inset benchmark`

---

### Task 10: Benchmark — PRML polynomial-fit small multiples

**Files:**
- Create: `figures/prml_polyfit_multiples.py`
- Baselines + readback record as usual

**Interfaces:** Consumes existing `Figure`/`Panel` grid, `Correspondence`, `plots.*`; exercises Task 4's gate across panels.

The claim (PRML Fig 1.4): model capacity sweeps from underfit to interpolation — same data, same true curve, only M varies. Content: 2×2 grid, M = 0, 1, 3, 9; in each panel the SAME 10-point dataset (fixed seed, sin(2πx) + noise computed in the program), the true curve, and the degree-M least-squares fit. Keys: `"data"`, `"truth"`, `"fit"` — identical hues everywhere (the fit keyed the same in all four panels IS the referential-noun discipline; the correspondence declares `varies="polynomial degree M"`, `changes=("fit",)`). Panel tags `[a]`–`[d]` with M stated per panel. Everything else pixel-identical across panels — that is the device, and `Correspondence` is what enforces it; assert the shared axis lims once.

- [ ] **Step 1: Write the program** (design steps 0–9 in the docstring)
- [ ] **Step 2: `make check` passes**
- [ ] **Step 3: `make regress` clean; `make update`**
- [ ] **Step 4: Readback record** written
- [ ] **Step 5: Commit** `figures: PRML 1.4 small multiples — swept-parameter family benchmark`

---

### Task 11: Docs — mark the layer landed, name the devices

**Files:**
- Modify: `docs/primitive-gaps.md` (move the landed items out of "Still missing", add a short "landed" note in the house style — measured, with the boundary each item found)
- Modify: `docs/grammar.md` (device entries: elision stack, declared truncation, unknown-mechanism node, operator junction, register channel, word-scale inset, annotated derivation, declared exaggeration-as-convention — each one paragraph, each pointing at its benchmark figure as the exemplar)
- Modify: `docs/skill.md` ONLY if it carries a device/primitive list that is now stale (read it; do not grow it — it is deliberately short)

No code. The bar: `docs/primitive-gaps.md` says the code wins when they disagree — make the docs agree with what Tasks 1–10 actually built, including any scope cuts the implementers recorded in their reports (the controller will provide the list of deviations).

- [ ] **Step 1: Update the three docs**
- [ ] **Step 2: `make test && make regress` still clean** (docs-only, but run it — cheap insurance)
- [ ] **Step 3: Commit** `docs: honesty marks, register, inset, derivation — landed; devices named`
