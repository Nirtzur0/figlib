"""The readback gate: the cold-reader test.

A fresh model — with NO access to the figure program, the claim, or any
conversation context — sees only the rendered PNG and answers "what does
this figure claim?" The figure passes iff the readback matches CLAIM.
This is the gate that catches caricature: geometry that looks like
mathematics but doesn't argue it.

This module holds the protocol (prompt + record format); actually
spawning the reader requires an agent harness, so the runner here only
prepares and records.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

READBACK_PROMPT = """\
Read the image file {png_path} and look at it carefully.

You are a cold reader: you have no other context about this figure.
Answer based ONLY on what the image shows:

1. In one sentence: what mathematical claim does this figure make?
   (Not a description of what's drawn — the claim the drawing is arguing for.)
2. In 2-3 bullets: anything that confused you, seemed ambiguous, or that
   you had to guess at.
"""


@dataclass
class ReadbackRecord:
    claim: str
    readback: str
    verdict: str        # 'pass' | 'fail'
    notes: str = ""

    def to_markdown(self) -> str:
        return (
            f"# Readback record\n\n"
            f"**Verdict:** {self.verdict}\n\n"
            f"## Intended claim\n\n{self.claim}\n\n"
            f"## Cold readback\n\n{self.readback}\n\n"
            f"## Notes\n\n{self.notes}\n"
        )


def prompt_for(png_path: str | Path) -> str:
    return READBACK_PROMPT.format(png_path=png_path)


def record(png_path: str | Path, rec: ReadbackRecord) -> Path:
    out = Path(png_path).with_suffix(".readback.md")
    out.write_text(rec.to_markdown())
    return out
