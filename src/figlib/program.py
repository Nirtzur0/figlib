"""The figure-program contract and runner.

A figure program is a module with:
    CLAIM: str            one-sentence claim the figure must communicate
    PARAMS: dict
    compute(params) -> geometry (any structure of arrays)
    build(geom) -> Scene
    assertions(geom)      numerical gate: raises AssertionError on failure

run() executes compute -> build -> gates -> render, writing SVG+PNG next
to the program under figures/out/.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

from .gates import Diagnostic, mechanical, numerical
from .render import save
from .style import DEFAULT_STYLE, Style


@dataclass
class Report:
    name: str
    claim: str
    svg_path: Path
    png_path: Path
    diagnostics: list[Diagnostic] = field(default_factory=list)

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
        style: Style = DEFAULT_STYLE, width_px: float = 900) -> Report:
    if not isinstance(program, ModuleType):
        program_path = Path(program)
        mod = load_program(program_path)
        default_out = program_path.parent / "out"
    else:
        mod = program
        default_out = Path(mod.__file__ or ".").parent / "out"
    out = Path(out_dir) if out_dir else default_out

    style = getattr(mod, "THEME", None) or style
    geom = mod.compute(mod.PARAMS)
    scene = mod.build(geom)
    diags = numerical(lambda: mod.assertions(geom))
    diags += mechanical(scene, style, width_px=width_px)
    svg_path, png_path = save(scene, out / mod.__name__, style, width_px=width_px)
    return Report(mod.__name__, mod.CLAIM, svg_path, png_path, diags)
