"""Transitions → Narsese: the percept vocabulary the driver feeds the mind.

The features are nars-memento's replay-evidence vocabulary (`frame_diff`,
`best_translation` — grid-derived only, no env internals), reprojected as
present-tense events (`:|:`) the zMind engine can ingest and revise:

    <(*,{a2}) --> [change]>.      :|:   something changed under ACTION2
    <(*,{a2}) --> [static]>.      :|:   nothing changed
    <(*,{a2},{9}) --> [appeared]>.:|:   symbol 9 showed up (first, row-major)
    <(*,{a2},{1}) --> [disappeared]>.:|:  symbol 1 vanished
    <(*,{a2}) --> [shift_dn2_3]>. :|:   a rigid shift (dx=-2, dy=3) explains
                                        the change at fit ≥ SHIFT_EMIT_FIT

Big-change gate: > BIG_CHANGE_CELLS cells also earns [big_change], so the
mind can separate level-scale redraws from local edits.

The operation itself is landed separately (`performed_event`, or the
act-resolution form for mind-proposed acts) — the motor report and the
percept batch stay distinct evidence streams.

Atom-word safety: Narsese words exclude `-` and most punctuation, so
negative deltas spell `n` (`dn2` = dx -2); op names are `[a1a7]|a6_x_y`.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

ZENGRAM_SRC = Path(
    os.environ.get(
        "ZENGRAM_SRC",
        Path(__file__).resolve().parents[2].parent / "zEngram" / "src",
    )
)
if str(ZENGRAM_SRC) not in sys.path:
    sys.path.insert(0, str(ZENGRAM_SRC))

try:
    # the zEngram rebrand (2026-11): the engine package's new name
    from zengram.nars_bridge.evidence import change_features  # noqa: E402
except ImportError:
    # pre-rebrand checkout (nars-memento): the bridge's evidence module
    # participates in an import cycle there (evidence → core.state →
    # core/__init__ → loop → evidence); importing the core package
    # first lets loop.py complete the cycle before we touch evidence
    # directly.
    _NARS_MEMENTO_SRC = Path(
        os.environ.get(
            "NARS_MEMENTO_SRC",
            ZENGRAM_SRC.parent / "nars-memento" / "src",
        )
    )
    if str(_NARS_MEMENTO_SRC) not in sys.path:
        sys.path.insert(0, str(_NARS_MEMENTO_SRC))
    import nars_memento.core  # noqa: F401, E402
    from nars_memento.nars_bridge.evidence import change_features  # noqa: E402

# A rigid translation must explain at least this fraction of the changed
# cells before the mind hears about a shift at all (below that the change
# is noise, not motion — honesty over coverage).
SHIFT_EMIT_FIT = 0.8
BIG_CHANGE_CELLS = 64  # ~1.5% of the 64×64 grid


def op_name(action: str) -> str:
    """`ACTION1` → `a1`; `ACTION6@16,32` → `a6_16_32` (Narsese-safe words)."""
    name, sep, coords = action.partition("@")
    if not name.startswith("ACTION"):
        raise ValueError(f"not an ARC action: {action!r}")
    num = name[len("ACTION"):]
    if not sep:
        return f"a{num}"
    x, _, y = coords.partition(",")
    int(x), int(y)  # validate shape; the server owns the ranges
    return f"a{num}_{x}_{y}"


def action_of_op(op: str) -> str:
    """The inverse of `op_name` (the act→env mapping)."""
    if not op.startswith("a"):
        raise ValueError(f"not a zAgent ARC op: {op!r}")
    body = op[1:]
    num, sep, coords = body.partition("_")
    if num not in {"1", "2", "3", "4", "5", "6", "7"}:
        raise ValueError(f"not a zAgent ARC op: {op!r}")
    if not sep:
        return f"ACTION{num}"
    x, _, y = coords.partition("_")
    return f"ACTION{num}@{x},{y}"


def _shift_token(dx: int, dy: int) -> str:
    def part(v: int) -> str:
        return f"n{-v}" if v < 0 else str(v)

    return f"shift_d{part(dx)}_{part(dy)}"


def performed_event(op: str) -> str:
    """The operation-was-performed landing, in the zmind_observe convention
    (the reporter sets f, the mind owns c at the system-one floor): the
    act-resolution form lands this for mind-proposed acts; the driver lands
    it directly for fallback actions so the evidence stream stays uniform."""
    return f"<(*,{{SELF}}) --> ^{op}>. :|: %1.00;0.20%"


@dataclass
class TransitionReport:
    """What one transition earns: the `:|:` lines plus the counts the run
    log records (the analysis reads these, the mind reads the lines)."""

    events: list[str]
    changed_cells: int
    appeared: str | None
    disappeared: str | None
    shift: str | None


def transition(op: str, prev_grid: list[list[str]], grid: list[list[str]]) -> TransitionReport:
    """Compile one performed transition: percept lines + the log summary.

    The features come from nars-memento's `change_features` — the same
    one-call derivation the replay gate's evidence accounting uses, so the
    mind's percepts and the harness's ledgers cannot drift."""
    diff = change_features(prev_grid, grid)
    events: list[str] = []
    shift: str | None = None
    if diff["changed"]:
        events.append(f"<(*,{{{op}}}) --> [change]>. :|:")
        if diff["changed_cells"] > BIG_CHANGE_CELLS:
            events.append(f"<(*,{{{op}}}) --> [big_change]>. :|:")
        for key, pred in (("appeared", "appeared"), ("disappeared", "disappeared")):
            sym = diff[key]
            if sym is not None:
                events.append(f"<(*,{{{op}}},{{{sym}}}) --> [{pred}]>. :|:")
        if diff["shift_fit"] >= SHIFT_EMIT_FIT:
            shift = _shift_token(diff["shift_dx"], diff["shift_dy"])
            events.append(f"<(*,{{{op}}}) --> [{shift}]>. :|:")
    else:
        events.append(f"<(*,{{{op}}}) --> [static]>. :|:")
    return TransitionReport(
        events=events,
        changed_cells=diff["changed_cells"],
        appeared=diff["appeared"],
        disappeared=diff["disappeared"],
        shift=shift,
    )


def boundary_events(arc3_state: str, leveled: bool) -> list[str]:
    """Episode-boundary events (level completion / WIN / GAME_OVER)."""
    events: list[str] = []
    if leveled:
        events.append("<(*,{SELF}) --> [level_up]>. :|:")
    if arc3_state == "WIN":
        events.append("<(*,{SELF}) --> [win]>. :|:")
    elif arc3_state == "GAME_OVER":
        events.append("<(*,{SELF}) --> [game_over]>. :|:")
    return events


def budget_event(step: int, ceiling: int) -> str:
    """The action-economy progress band (the arc3 analogue of the
    distance bands the other venues needed: the win-enabling minimum's
    approach/progress signal). Where the episode stands against the
    objective's pre-registered action ceiling — how much of the budget
    a clear is still allowed to cost. Mutually exclusive per step;
    render-side only (the seam invariant: percepts, never beliefs)."""
    if step >= ceiling * 2 // 3:
        band = "budget_late"
    elif step >= ceiling // 3:
        band = "budget_mid"
    else:
        band = "budget_early"
    return f"<(*,{{SELF}}) --> [{band}]>. :|:"


def change_probe(ops: list[str]) -> list[tuple[str, str]]:
    """The per-op change-maker questions: `does op leave the frame changed?`

    Answers (where the mind has any) arrive as `answer:` lines with truth
    values — the advisory signal the post-run analysis compares against the
    observed per-op change rates."""
    return [(op, f"<(*,{{{op}}}) --> [change]>?") for op in ops]
