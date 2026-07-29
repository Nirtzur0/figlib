"""Golden-figure regression: does the renderer still draw the committed corpus?

The unit of comparison is the SVG *text* a figure program emits. Rendering
is a pure function of the program (fixed grain seed, no wall-clock, no
hashing of object ids into the output), so byte equality is the correct
fast path — not a fuzzy match.

When the text differs, byte equality says nothing about how much moved. So
we rasterize both SVGs through the same cairosvg call and report two
numbers: the fraction of pixels that changed at all, and the RMS channel
difference. A nudged label lands near 0.1% changed; a redrawn curve family
lands in the percents. That separation is the whole point of the metric —
it lets a human decide "cosmetic, accept" vs "structural, look at it".

    compare_figure(path, out_dir) -> RegressResult(status=match|drift|new|error)

Nothing here writes into the baseline directory. `update_baselines` is the
only function that does, and it is a separate, explicit call.
"""

from __future__ import annotations

import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: max per-channel difference (0-255) still counted as "the same pixel"
PIXEL_TOL = 2
#: raster scale for the drift metric; 1.0 keeps the sweep fast and is
#: enough resolution to separate cosmetic from structural change
METRIC_SCALE = 1.0

MATCH, DRIFT, NEW, ERROR = "match", "drift", "new", "error"


@dataclass(frozen=True)
class RegressResult:
    name: str
    status: str
    detail: str = ""
    changed_fraction: float = 0.0
    rms: float = 0.0

    @property
    def ok(self) -> bool:
        """New baselines are not a regression; drift and errors are."""
        return self.status in (MATCH, NEW)

    def line(self) -> str:
        metric = ""
        if self.status == DRIFT:
            metric = f"{100 * self.changed_fraction:6.2f}% px  rms {self.rms:5.1f}"
        return f"{self.status.upper():<6} {self.name:<34} {metric:<24} {self.detail}".rstrip()


def discover_programs(figures_dir: str | Path) -> list[Path]:
    """Every figure program under figures_dir: *.py, no leading underscore,
    top level only (figures/out/ holds artifacts, not programs)."""
    return sorted(p for p in Path(figures_dir).glob("*.py")
                  if not p.name.startswith("_"))


def _render_to(program_path: Path, dest: Path, transparent: bool) -> Path:
    """Run the program into `dest`; return the SVG path. Raises on a failed
    gate so a broken figure cannot masquerade as a clean match."""
    from .program import run

    report = run(program_path, out_dir=dest, transparent=transparent)
    if not report.passed:
        kinds = "; ".join(f"{d.kind}: {d.detail}" for d in report.diagnostics)
        raise RuntimeError(f"gates failed -- {kinds}")
    return report.svg_path


def _raster(svg_path: Path, scale: float = METRIC_SCALE) -> np.ndarray:
    import cairosvg
    from PIL import Image

    with tempfile.TemporaryDirectory() as td:
        png = Path(td) / "r.png"
        cairosvg.svg2png(url=str(svg_path), write_to=str(png), scale=scale)
        with Image.open(png) as im:
            return np.asarray(im.convert("RGBA"), dtype=np.int16)


def _pixel_metrics(baseline_svg: Path, fresh_svg: Path) -> tuple[float, float, str]:
    """(changed_fraction, rms, note). Differently sized canvases are 100%
    changed by definition -- the figure's Format moved."""
    a, b = _raster(baseline_svg), _raster(fresh_svg)
    if a.shape != b.shape:
        note = (f"canvas {a.shape[1]}x{a.shape[0]} -> {b.shape[1]}x{b.shape[0]}")
        return 1.0, 255.0, note
    diff = np.abs(a - b)
    changed = float(np.mean(diff.max(axis=2) > PIXEL_TOL))
    rms = float(np.sqrt(np.mean(diff.astype(np.float64) ** 2)))
    return changed, rms, ""


def compare_figure(program_path: str | Path, out_dir: str | Path,
                   transparent: bool = False) -> RegressResult:
    """Render `program_path` to scratch and compare against the committed
    SVG in `out_dir`. Pure with respect to `out_dir`: nothing is written."""
    program_path = Path(program_path)
    out_dir = Path(out_dir)
    name = program_path.stem + ("_transparent" if transparent else "")
    baseline = out_dir / f"{name}.svg"

    with tempfile.TemporaryDirectory() as td:
        try:
            fresh = _render_to(program_path, Path(td), transparent)
        except Exception:
            lines = traceback.format_exc().strip().splitlines()
            tail = " | ".join(ln.strip() for ln in lines[-2:])
            return RegressResult(name, ERROR, tail)

        if not baseline.exists():
            return RegressResult(name, NEW, f"no baseline at {baseline}")

        fresh_text = fresh.read_text()
        if fresh_text == baseline.read_text():
            return RegressResult(name, MATCH)

        changed, rms, note = _pixel_metrics(baseline, fresh)
        detail = note or f"svg {len(baseline.read_text())} -> {len(fresh_text)} chars"
        return RegressResult(name, DRIFT, detail, changed, rms)


def variants(program_path: Path, out_dir: Path) -> list[bool]:
    """Which renders the corpus commits for this program: always the plain
    one, plus the transparent one iff a transparent baseline exists."""
    if (out_dir / f"{program_path.stem}_transparent.svg").exists():
        return [False, True]
    return [False]


def sweep(figures_dir: str | Path, out_dir: str | Path | None = None,
          programs: list[Path] | None = None) -> list[RegressResult]:
    figures_dir = Path(figures_dir)
    out = Path(out_dir) if out_dir else figures_dir / "out"
    paths = programs if programs is not None else discover_programs(figures_dir)
    return [compare_figure(p, out, transparent=t)
            for p in paths for t in variants(p, out)]


def update_baselines(programs: list[Path], out_dir: str | Path) -> list[Path]:
    """Overwrite the committed SVG+PNG for each program. Returns SVG paths."""
    out = Path(out_dir)
    written: list[Path] = []
    for p in programs:
        for t in variants(p, out):
            written.append(_render_to(p, out, t))
    return written
