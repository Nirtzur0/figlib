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
    ap.add_argument("--width", type=float, default=1100)
    ap.add_argument("--readback-prompt", action="store_true",
                    help="print the cold-reader prompt for the rendered PNG")
    args = ap.parse_args(argv)

    report = run(args.program, width_px=args.width)
    print(report.summary())
    if args.readback_prompt:
        print("\n--- readback prompt ---")
        print(prompt_for(report.png_path))
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
