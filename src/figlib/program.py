"""The figure-program contract and runner.

A figure program is a module with:
    CLAIM: str            one-sentence claim the figure must communicate
    PARAMS: dict
    compute(params) -> geometry (any structure of arrays)
    build(geom) -> Scene | Figure (multi-panel)
    assertions(geom)      numerical gate: raises AssertionError on failure

run() executes compute -> build -> autoplace -> gates -> render, writing
SVG+PNG next to the program under figures/out/. The auto-place pass
(autoplace.py) resolves label collisions/clipping it can fix within
budget before the gates measure; gates and render then see the same
solved scene.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

from .autoplace import autoplace, autoplace_figure
from .figure import Figure
from .gates import (Diagnostic, color_gate, color_gate_figure, mechanical,
                    mechanical_figure, numerical)
from .render import save, save_figure
from .style import DEFAULT_STYLE, Style


@dataclass
class Report:
    name: str
    claim: str
    svg_path: Path
    png_path: Path
    diagnostics: list[Diagnostic] = field(default_factory=list)
    # what build() returned plus the resolved style/width, so callers
    # (figcheck --report) can inspect layout without re-running the program
    built: object = None
    style: Style | None = None
    width_px: float | None = None
    # informational, not failures: the derived format, auto-place moves
    notes: list[str] = field(default_factory=list)
    # floor-side expressivity signals (expressivity.py) — advisory, so the
    # iteration loop sees under-expression; sparseness never fails a gate
    signals: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.diagnostics

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"[{status}] {self.name}: {self.claim}", f"  svg: {self.svg_path}", f"  png: {self.png_path}"]
        lines += [f"  {n}" for n in self.notes]
        lines += [f"  {d.kind}: {d.detail}" for d in self.diagnostics]
        # signals are self-labelling ("expressivity: ...", "clearance: ...")
        lines += [f"  {s}" for s in self.signals]
        return "\n".join(lines)


def load_program(path: str | Path) -> ModuleType:
    """Load a figure program from source, always.

    A .pyc is validated against (int(source_mtime), source_size), so an edit
    that keeps the byte count and lands in the same second silently serves
    stale bytecode -- and a regression sweep then reports a false MATCH.
    Figure programs execute once per run, so the cache buys nothing: compile
    the source text directly and never touch __pycache__.
    """
    path = Path(path)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader, f"cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    exec(compile(path.read_text(), str(path), "exec"), mod.__dict__)
    for attr in ("CLAIM", "PARAMS", "compute", "build", "assertions"):
        assert hasattr(mod, attr), f"figure program {path.name} missing {attr}"
    return mod


def run(program: str | Path | ModuleType, out_dir: str | Path | None = None,
        style: Style = DEFAULT_STYLE, width_px: float | None = None,
        transparent: bool = False, place: bool = True) -> Report:
    if not isinstance(program, ModuleType):
        program_path = Path(program)
        mod = load_program(program_path)
        default_out = program_path.parent / "out"
    else:
        mod = program
        default_out = Path(mod.__file__ or ".").parent / "out"
    out = Path(out_dir) if out_dir else default_out

    geom = mod.compute(mod.PARAMS)
    built = mod.build(geom)          # a Scene or a multi-panel Figure

    # Format: declared = the page slot (an external constraint, policed);
    # undeclared = a free variable, derived from the annotation census.
    from .format import FORMATS, derive_format, smallest_carrier
    base_style = getattr(mod, "THEME", None) or style
    fmt = getattr(mod, "FORMAT", None)
    notes: list[str] = []
    if fmt is None:
        fmt = derive_format(built, base_style)
        notes.append(f"format: {fmt.name} (derived from annotation census)")
    if width_px is None:
        width_px = fmt.display_width_px
    style = base_style.scaled(fmt.ink_scale)
    if transparent:
        from .theme import Theme, transparent_variant
        if isinstance(style, Theme):
            style = transparent_variant(style)
    diags = numerical(lambda: mod.assertions(geom))
    stem = mod.__name__ + ("_transparent" if transparent else "")
    if isinstance(built, Figure):
        if place:
            notes += [f"placed: {n}" for n in
                      autoplace_figure(built, style, width_px=width_px)]
        diags += mechanical_figure(built, style, width_px=width_px)
        diags += color_gate_figure(built, style)
        # Hue is a referential noun: pooled across panels, no declaration
        # needed — a hue that names two things is broken whether or not the
        # figure claims a correspondence.
        from .correspond import hue_binding_violations
        diags += hue_binding_violations([p.scene for p in built.panels], style)
        # Composite figures only: the residual against a declared binding.
        # Silent when a program declares none, so a figure whose parts are
        # independent is not forced to invent a relation between them.
        corrs = getattr(mod, "CORRESPONDENCE", None)
        if corrs:
            from .correspond import residual
            diags += residual(built, corrs, style, width_px=width_px)
        svg_path, png_path = save_figure(built, out / stem, style, width_px=width_px)
    else:
        if place:
            notes += [f"placed: {n}" for n in
                      autoplace(built, style, width_px=width_px)]
        diags += mechanical(built, style, width_px=width_px)
        diags += color_gate(built, style)
        from .correspond import hue_binding_violations
        diags += hue_binding_violations(built, style)
        svg_path, png_path = save(built, out / stem, style, width_px=width_px)

    # The load gate's advice is computed, not asserted: evaluate whether a
    # larger format actually carries the census (WIDE's ink_scale cancels
    # most of its extra canvas) before recommending one.
    load_kinds = ("annotation-load", "label-scale")
    if any(d.kind in load_kinds for d in diags):
        bigger = tuple(f for f in FORMATS
                       if f.display_width_px > fmt.display_width_px)
        carrier = smallest_carrier(built, base_style, ladder=bigger) if bigger else None
        hint = (f"FORMAT = {carrier.name.upper()} would carry this load"
                if carrier else "no format carries this load — trim annotation")
        diags = [Diagnostic(d.kind, f"{d.detail} [computed: {hint}]")
                 if d.kind in load_kinds else d for d in diags]
    from .expressivity import signals as expressivity_signals
    from .place import clearance_signal
    sigs = [f"expressivity: {s}"
            for s in expressivity_signals(built, style, width_px)]
    # "does this label touch that stroke?" — asked by 212 PNG looks across
    # the corpus, answered by a number. Rides every run, not a flag.
    sigs += clearance_signal(built, style, width_px)
    return Report(mod.__name__, mod.CLAIM, svg_path, png_path, diags,
                  built=built, style=style, width_px=width_px, notes=notes,
                  signals=sigs)
