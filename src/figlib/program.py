"""The figure-program contract and runner.

A figure program is a module with:
    CLAIM: str            one-sentence claim the figure must communicate
    PARAMS: dict
    compute(params) -> geometry (any structure of arrays)
    build(geom) -> Scene | Figure (multi-panel)
    assertions(geom)      numerical gate: raises AssertionError on failure

run() executes compute -> build -> gates -> render, writing SVG+PNG next
to the program under figures/out/.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

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

    @property
    def passed(self) -> bool:
        return not self.diagnostics

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"[{status}] {self.name}: {self.claim}", f"  svg: {self.svg_path}", f"  png: {self.png_path}"]
        lines += [f"  {d.kind}: {d.detail}" for d in self.diagnostics]
        return "\n".join(lines)


def load_program(path: str | Path) -> ModuleType:
    path = Path(path)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader, f"cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for attr in ("CLAIM", "PARAMS", "compute", "build", "assertions"):
        assert hasattr(mod, attr), f"figure program {path.name} missing {attr}"
    return mod


def run(program: str | Path | ModuleType, out_dir: str | Path | None = None,
        style: Style = DEFAULT_STYLE, width_px: float | None = None,
        transparent: bool = False) -> Report:
    if not isinstance(program, ModuleType):
        program_path = Path(program)
        mod = load_program(program_path)
        default_out = program_path.parent / "out"
    else:
        mod = program
        default_out = Path(mod.__file__ or ".").parent / "out"
    out = Path(out_dir) if out_dir else default_out

    from .format import COLUMN
    fmt = getattr(mod, "FORMAT", COLUMN)
    if width_px is None:
        width_px = fmt.display_width_px
    style = getattr(mod, "THEME", None) or style
    style = style.scaled(fmt.ink_scale)
    if transparent:
        from .theme import Theme, transparent_variant
        if isinstance(style, Theme):
            style = transparent_variant(style)
    geom = mod.compute(mod.PARAMS)
    built = mod.build(geom)          # a Scene or a multi-panel Figure
    diags = numerical(lambda: mod.assertions(geom))
    stem = mod.__name__ + ("_transparent" if transparent else "")
    if isinstance(built, Figure):
        diags += mechanical_figure(built, style, width_px=width_px)
        diags += color_gate_figure(built, style)
        svg_path, png_path = save_figure(built, out / stem, style, width_px=width_px)
    else:
        diags += mechanical(built, style, width_px=width_px)
        diags += color_gate(built, style)
        svg_path, png_path = save(built, out / stem, style, width_px=width_px)
    return Report(mod.__name__, mod.CLAIM, svg_path, png_path, diags,
                  built=built, style=style, width_px=width_px)
