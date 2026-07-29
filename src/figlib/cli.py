"""figcheck: run a figure program through render + deterministic gates.

    figcheck figures/fig09_exp_series_spiral.py [--width 1100]

Exit code 1 if any gate fails. The readback gate needs an agent harness;
figcheck prints the prompt to hand to one.
"""

from __future__ import annotations

import argparse
import sys

from .program import run
from .readback import prompt_for


def _zoom_main(report, spec: str) -> None:
    """--zoom x0,y0,x1,y1[:scale]: crop the rendered PNG around a math-coords
    window (figure canvas px for multi-panel figures, whose math coords are
    panel-local) and write it as <stem>.zoom.png. Replaces the hand-written
    PIL crop dance."""
    from PIL import Image

    from .layout import Transform
    from .render import PNG_SCALE
    from .scene import Scene

    scale = 1
    if ":" in spec:
        spec, s = spec.rsplit(":", 1)
        scale = max(1, int(s))
    x0, y0, x1, y1 = (float(v) for v in spec.split(","))
    if isinstance(report.built, Scene):
        t = Transform(report.built, width_px=report.width_px)
        (cx0, cy0), (cx1, cy1) = t.to_canvas((x0, y1)), t.to_canvas((x1, y0))
    else:  # Figure: coords are already figure canvas px (+y down)
        (cx0, cy0), (cx1, cy1) = (x0, y0), (x1, y1)
    with Image.open(report.png_path) as im:
        box = (max(int(cx0 * PNG_SCALE), 0), max(int(cy0 * PNG_SCALE), 0),
               min(int(cx1 * PNG_SCALE + 0.5), im.width),
               min(int(cy1 * PNG_SCALE + 0.5), im.height))
        if box[0] >= box[2] or box[1] >= box[3]:
            print(f"zoom: window {spec} maps to an empty crop {box}")
            return
        crop = im.crop(box)
        if scale > 1:
            crop = crop.resize((crop.width * scale, crop.height * scale),
                               Image.NEAREST)
        out = report.png_path.with_name(report.png_path.stem + ".zoom.png")
        crop.save(out)
    print(f"zoom: {out}")


def _regress_main(args) -> int:
    """--regress / --update: the golden-figure harness over the corpus."""
    from pathlib import Path

    from .regress import discover_programs, sweep, update_baselines

    figures_dir = Path(args.figures_dir)
    named = [Path(p) for p in args.program]
    programs = named or discover_programs(figures_dir)

    if args.update:
        written = update_baselines(programs, figures_dir=figures_dir)
        for p in written:
            print(f"UPDATED {p}")
        print(f"\n{len(written)} baseline(s) refreshed under {figures_dir / 'out'}")
        return 0

    results = sweep(figures_dir, programs=programs)
    for r in results:
        print(r.line())
    bad = [r for r in results if not r.ok]
    tally = {k: sum(r.status == k for r in results)
             for k in ("match", "drift", "new", "error")}
    print("\n" + "  ".join(f"{k}={v}" for k, v in tally.items()))
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="figcheck")
    ap.add_argument("program", nargs="*",
                    help="path to a figure program .py (exactly one, unless "
                         "--regress/--update, where an empty list means all)")
    ap.add_argument("--width", type=float, default=None,
                    help="override the figure's declared FORMAT width (CSS px)")
    ap.add_argument("--readback-prompt", action="store_true",
                    help="print the cold-reader prompt for the rendered PNG")
    ap.add_argument("--paper", action="store_true",
                    help="render the theme's paper ground, grain and casings; "
                         "the default is no ground (SVG+PNG keep alpha)")
    ap.add_argument("--transparent", action="store_true",
                    help=argparse.SUPPRESS)   # deprecated: now the default
    ap.add_argument("--no-autoplace", action="store_true",
                    help="skip the label auto-place pass (raw declared offsets)")
    ap.add_argument("--report", action="store_true",
                    help="print a textual scene inventory (label bboxes, "
                         "geometry extents, margins) for layout debugging")
    ap.add_argument("--zoom", metavar="X0,Y0,X1,Y1[:SCALE]", default=None,
                    help="write a crop of the rendered PNG as <stem>.zoom.png; "
                         "coords are MATH coords for a single scene, figure "
                         "canvas px for a multi-panel figure; :SCALE upsamples "
                         "the crop (nearest) for inspection")
    ap.add_argument("--gallery", action="store_true",
                    help="regenerate figures/out/GALLERY.md from program metadata")
    ap.add_argument("--regress", action="store_true",
                    help="sweep the corpus: re-render every figure and diff "
                         "against the committed SVG; nonzero on drift/error")
    ap.add_argument("--update", action="store_true",
                    help="overwrite the committed SVG+PNG baselines")
    ap.add_argument("--figures-dir", default="figures",
                    help="corpus root for --regress/--update (default: figures)")
    args = ap.parse_args(argv)

    if args.gallery:
        from pathlib import Path

        from .gallery import write_gallery
        print(f"gallery: {write_gallery(Path(args.figures_dir))}")
        return 0
    if args.regress or args.update:
        return _regress_main(args)
    if len(args.program) != 1:
        ap.error("exactly one figure program is required")

    # Artifacts land in the mirrored out/ tree, exactly where --regress looks
    # for them. Without this, `program.run` falls back to <program>/../out, so
    # a subject-binned figure would write figures/complex/out/ and the sweep
    # would never see it -- a single render silently disagreeing with the
    # corpus about where its own baseline lives.
    from pathlib import Path

    from .regress import artifact_dir

    # Only when the program really lives under --figures-dir; a program given
    # by absolute path from somewhere else keeps program.run's own default
    # (<program>/../out), which is what host projects outside this repo rely on.
    program = Path(args.program[0])
    figures_dir = Path(args.figures_dir)
    out_dir = None
    try:
        program.resolve().relative_to(figures_dir.resolve())
        out_dir = artifact_dir(program, figures_dir)
    except ValueError:
        pass
    report = run(program, out_dir=out_dir, width_px=args.width,
                 paper=args.paper, place=not args.no_autoplace)
    print(report.summary())
    if args.report:
        from .report import report as layout_report
        print("\n--- layout report ---")
        print(layout_report(report.built, report.style, report.width_px))
    if args.readback_prompt:
        print("\n--- readback prompt ---")
        print(prompt_for(report.png_path, report.built))
    if args.zoom:
        _zoom_main(report, args.zoom)
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
