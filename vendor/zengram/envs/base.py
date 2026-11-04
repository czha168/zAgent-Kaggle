"""Environment protocol: structured, deterministic observations.

Envs return full observations (symbol grid + goal flag) so the replay verifier
(Phase 1) can compare every cell of every frame without touching the live env
(proposal §4.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class Obs:
    grid: list[list[str]]
    goal_reached: bool
    steps: int
    info: dict | None = None


class Env(Protocol):
    actions: list[str]

    def reset(self) -> Obs: ...
    def step(self, action: str) -> Obs: ...


def render(obs: Obs) -> str:
    return "\n".join("".join(row) for row in obs.grid)


def grids_equal(a: list[list[str]], b: list[list[str]]) -> bool:
    return a == b


def grid_cell_stats(a: list[list[str]], b: list[list[str]]) -> tuple[int, int]:
    """`(total, differing)` cell counts — the one definition of grid
    difference (mechanical-correction-plan Phase F): the verify gate's cell
    accounting, the planning comparator's equality, and the anticipation
    channel's error fraction all derive from this, so they cannot drift.

    Ragged or extra rows and columns count as differing cells; a
    structural difference carrying no cells at all (e.g. `[[]]` vs `[]`)
    counts as one differing cell, so `differing == 0` if and only if the
    grids are exactly equal."""
    if a == b:
        return sum(len(row) for row in a), 0
    rows = max(len(a), len(b))
    total = differ = 0
    for r in range(rows):
        row_a = a[r] if r < len(a) else []
        row_b = b[r] if r < len(b) else []
        cols = max(len(row_a), len(row_b))
        total += cols
        differ += sum(
            1 for c in range(cols)
            if (row_a[c] if c < len(row_a) else None)
            != (row_b[c] if c < len(row_b) else None)
        )
    if total == 0:
        return 1, 1
    return total, differ
