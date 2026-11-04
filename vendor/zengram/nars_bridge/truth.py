"""NAL truth values and the revision function (proposal §10, row 1).

A rule's truth is Wang's Non-Axiomatic Logic pair `(f, c)`:

- `f` — frequency: the fraction of known evidence supporting the statement.
- `c` — confidence: the proportion of *future* evidence already accounted
  for, `c = w / (w + k)` over the accumulated evidence amount `w` (k = 1).

Two operations carry the bridge:

- `from_evidence(pos, neg)` — replay transitions that confirm/refute a rule
  map directly onto evidence amounts (`w = pos + neg`).
- `revise(a, b)` — NAL-6 revision: evidence-weighted frequency, additive
  evidence (`w1 + w2`). This is the arbitration mechanism §10 gives to the
  model population: competing statements combine instead of one side winning.

Confidence is clamped strictly below 1.0 at construction so `w = c/(1−c)`
stays finite; the sync operator clamps LLM-provided seeds before storing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

K = 1.0  # NAL's evidence-constant k: one unseen piece of evidence
_MAX_CONFIDENCE = 1.0 - 1e-6


@dataclass(frozen=True)
class TruthValue:
    f: float  # frequency in [0, 1]
    c: float  # confidence in [0, 1)

    def __post_init__(self):
        if not (0.0 <= self.f <= 1.0):
            raise ValueError(f"frequency must be in [0, 1], got {self.f}")
        if not (0.0 <= self.c < 1.0):
            raise ValueError(f"confidence must be in [0, 1), got {self.c}")

    @property
    def evidence(self) -> float:
        """w = c/(1−c): how much evidence c already represents."""
        return self.c / (1.0 - self.c)

    def revise(self, other: "TruthValue") -> "TruthValue":
        """NAL-6 revision of two judgments about the same statement."""
        w1 = self.evidence
        w2 = other.evidence
        w = w1 + w2
        if w <= 0.0:
            return TruthValue(0.5, 0.0)
        f = (w1 * self.f + w2 * other.f) / w
        return TruthValue(f, w / (w + K))

    def expectation(self) -> float:
        """Degree of belief used for ranking: c·(f − 0.5) + 0.5."""
        return expectation(self)


def expectation(tv: TruthValue) -> float:
    return tv.c * (tv.f - 0.5) + 0.5


def from_evidence(pos: int, neg: int) -> TruthValue:
    """Convert confirming/refuting transition counts into a truth value.

    Evidence amounts add directly (`w = pos + neg`), so this composes with
    `revise`: a seed judgment revised with accumulated evidence behaves as if
    the seed were its own piece of evidence."""
    total = pos + neg
    if total == 0:
        return TruthValue(0.5, 0.0)
    return TruthValue(pos / total, total / (total + K))


def clamped(f: float, c: float) -> TruthValue:
    """Sanitize an externally supplied (f, c) — e.g. an LLM-declared seed —
    into the valid domain instead of rejecting the whole record list."""
    if not (math.isfinite(f) and math.isfinite(c)):
        return TruthValue(0.5, 0.0)
    return TruthValue(
        min(1.0, max(0.0, float(f))),
        min(_MAX_CONFIDENCE, max(0.0, float(c))),
    )
