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
Read the image file {png_path}.

You are a cold reader: you have no other context about this figure.
Answer based ONLY on what the image shows, in two passes:

GLANCE (before studying anything): from your first overall impression
only — the largest shapes, the most prominent marks —
1. In one phrase: what is this figure about, and what does it seem to
   be asserting?

STUDY (now look carefully):
2. In one sentence: what mathematical claim does this figure make?
   (Not a description of what's drawn — the claim the drawing is arguing for.)
3. In 2-3 bullets: anything that confused you, seemed ambiguous, or that
   you had to guess at.
4. Which parts of the argument could you verify by inspection of the
   figure itself, and which did you have to take on trust?

The GLANCE answer is graded separately: a figure whose glance read
misses the claim has a macro-structure failure even if the studied
read recovers it.
"""

# Asked only of composites. The residual check (correspond.py) measures
# the differences it can see in the scene; this measures the ones a reader
# sees, which is the actual definition of the figure failing to click.
# A second difference named here that the program did not declare is the
# residual leaking through a channel no gate reaches.
BINDING_QUESTIONS = """\

BINDING (this figure has {n} parts):
5. What is held the SAME across the parts — which marks are the same
   object seen more than once?
6. What is DIFFERENT? List every difference you notice, not just the one
   that seems intended.
7. Which single difference does the figure appear to be arguing about?
"""


def _n_parts(built: object) -> int:
    panels = getattr(built, "panels", None)
    return len(panels) if panels else 0


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


def prompt_for(png_path: str | Path, built: object = None) -> str:
    """The cold-reader prompt, plus the binding questions when the figure
    is a composite — the only check on `Correspondence.varies`, which is
    prose and therefore unreachable by any deterministic gate."""
    prompt = READBACK_PROMPT.format(png_path=png_path)
    n = _n_parts(built)
    return prompt + BINDING_QUESTIONS.format(n=n) if n > 1 else prompt


def record(png_path: str | Path, rec: ReadbackRecord) -> Path:
    out = Path(png_path).with_suffix(".readback.md")
    out.write_text(rec.to_markdown())
    return out
