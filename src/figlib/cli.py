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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="figcheck")
    ap.add_argument("program", help="path to a figure program .py")
    ap.add_argument("--width", type=float, default=None,
                    help="override the figure's declared FORMAT width (CSS px)")
    ap.add_argument("--readback-prompt", action="store_true",
                    help="print the cold-reader prompt for the rendered PNG")
    ap.add_argument("--transparent", action="store_true",
                    help="render with no paper/grain; SVG+PNG keep alpha")
    ap.add_argument("--report", action="store_true",
                    help="print a textual scene inventory (label bboxes, "
                         "geometry extents, margins) for layout debugging")
    args = ap.parse_args(argv)

    report = run(args.program, width_px=args.width, transparent=args.transparent)
    print(report.summary())
    if args.report:
        from .report import report as layout_report
        print("\n--- layout report ---")
        print(layout_report(report.built, report.style, report.width_px))
    if args.readback_prompt:
        print("\n--- readback prompt ---")
        print(prompt_for(report.png_path))
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
