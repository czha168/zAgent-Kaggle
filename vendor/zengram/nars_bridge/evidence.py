"""Replay evidence → NARS judgments (proposal §10, row 2).

A replay counterexample is negative evidence; a confirming transition is
positive evidence. This module makes that attribution *mechanical*: every
recorded transition is compiled into a small feature vector derived only
from the observation grids (never env internals), and each rule may declare
an attribution `pattern` — a trigger plus a predicted outcome over those
features. A transition whose trigger matches and whose outcome matches
confirms the rule; a matched trigger with a mismatched outcome refutes it.

The feature vocabulary is env-generic for single-agent grid envs where the
agent renders as 'A' (the gridworld suite; ARC's multi-cell agents are out
of the prototype's scope — documented in docs/NARS_BRIDGE.md). `held_keys`
is derived per episode from the ground-key/door deltas, which is exact for
the pick-up/consume semantics of the gridworld suite.

Scanning is a full idempotent rescan of the history, so counts can never
drift from the record.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..envs.base import Obs
from .rules import (
    CONFIRMED_BY_CAP,
    FALSIFIED_BY_CAP,
    FATE_ACTIVE,
    FATE_REFUTED,
    RuleStore,
)

if TYPE_CHECKING:  # annotation-only: core and this package mutually
    from ..core.state import History  # import; neither may trigger the other's init

DIRECTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

FEATURE_KEYS = (
    # -- agent-centric (single-agent grid envs where the agent renders as 'A';
    #    inert on frames without an 'A' cell, e.g. ARC) --
    "action",
    "target_symbol",
    "moved",
    "target_symbol_after",
    "old_cell_symbol_after",
    "keys_delta",
    "doors_delta",
    "held_keys",
    # -- episode-position (beyond-llm plan R4 vocabulary extension): the
    #    host bookkeeping the record itself carries — steps since the
    #    episode's start. Not a grid cell, but recorded per transition, so
    #    it is not env-internal state; it is how time-shaped mechanics
    #    (periodic goals, expiry, lever sequences) become pattern-
    #    expressible. None when a scan does not supply it (such transitions
    #    can never match an episode_t-keyed trigger — honest absence). --
    "episode_t",
    # -- progress position (the corpus study's LEVEL discipline): the
    #    env-recorded level/stage the successor frame belongs to, read from
    #    obs_next.info when the env records one. Not a learnable cell (the
    #    venue barrier keeps score/progress out of the grids) but host
    #    bookkeeping like episode_t — how per-level mechanics (per-level
    #    budgets, level-gated mechanics, AWM's per-level concepts, each
    #    scoped "换关返回 NA" — level-change returns NA) become pattern-
    #    expressible. None when the env records no level (every current
    #    env): a level-keyed trigger can never match — honest absence. --
    "level",
    # -- frame-diff (any grid environment) --
    "changed",
    "changed_cells",
    "appeared",
    "disappeared",
    "appeared_count",
    "disappeared_count",
    "goal_reached",
    # -- translation (frame-diff family, pp-closure-plan Phase B): the rigid
    #    translation best explaining the frame change — the vocabulary ARC's
    #    block-motion rules need (an agent-less rigid block translating by a
    #    fixed offset is not expressible via agent-centric keys) --
    "shift_dx",
    "shift_dy",
    "shift_fit",
)


def _find_agent(grid: list[list[str]]) -> tuple[int, int] | None:
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            if ch == "A":
                return (r, c)
    return None


def _cell(grid: list[list[str]], pos: tuple[int, int]) -> str | None:
    r, c = pos
    if 0 <= r < len(grid) and 0 <= c < len(grid[r]):
        return grid[r][c]
    return None


def _count(grid: list[list[str]], symbol: str) -> int:
    return sum(row.count(symbol) for row in grid)


# Agent-centric feature keys: on frames without an 'A' cell they are
# vacuous (always False/None), so a pattern keyed on them can never earn
# evidence there — such transitions count as no-evidence, never refutation.
_AGENT_CENTRIC_KEYS = frozenset(
    {
        "target_symbol",
        "moved",
        "target_symbol_after",
        "old_cell_symbol_after",
        "keys_delta",
        "doors_delta",
        "held_keys",
    }
)


def _frame_diff(grid: list[list[str]], grid2: list[list[str]]) -> dict:
    """Frame-diff features: what a transition changed, without any notion of
    an agent — the vocabulary that works on ARC frames (multi-cell entities,
    no distinguished 'A' cell). `appeared`/`disappeared` are the first new /
    vanished symbol in row-major scan order; the *_count variants count
    distinct such symbols so patterns can require a unique change."""
    old_symbols: set[str] = set()
    new_symbols: set[str] = set()
    changed = 0
    for r, row in enumerate(grid):
        row2 = grid2[r] if r < len(grid2) else []
        for c, ch in enumerate(row):
            old_symbols.add(ch)
            if c < len(row2):
                new_symbols.add(row2[c])
                if row2[c] != ch:
                    changed += 1
    extra = {c for row2 in grid2[len(grid):] for c in row2}
    new_symbols |= extra
    appeared_set = new_symbols - old_symbols
    disappeared_set = old_symbols - new_symbols

    def _first_present(symbol_set: set[str], grid_rows) -> str | None:
        if not symbol_set:
            return None
        for rows in grid_rows:
            for row in rows:
                for ch in row:
                    if ch in symbol_set:
                        return ch
        return None

    return {
        "changed": changed > 0,
        "changed_cells": changed,
        "appeared": _first_present(appeared_set, (grid2,)),
        "disappeared": _first_present(disappeared_set, (grid,)),
        "appeared_count": len(appeared_set),
        "disappeared_count": len(disappeared_set),
    }


# Matched-filter offset bound: translations beyond ±16 cells per axis are
# not searched (a bounded, deterministic scan; real block motion is local).
SHIFT_WINDOW = 16


def best_translation(grid: list[list[str]], grid2: list[list[str]]) -> tuple[int, int, float]:
    """The rigid translation best explaining the frame change: `(dx, dy, fit)`.

    `dx` rightward / `dy` downward in cells (negative = left / up), full-grid
    coordinates. Scored on the changed cells only — static cells match
    trivially under any offset, so they carry no information: a changed cell
    (r, c) is *explained* by offset (dr, dc) when grid2[r+dr][c+dc] equals
    grid[r][c] (out-of-bounds never matches; a vacated cell's symbol is found
    at its translated position, and an appearing cell's old symbol beyond
    the moved content matches too). `fit` is the explained fraction of
    changed cells — the honesty gauge: `shift_dx/dy` of a change that is no
    translation at all (fit ~0) is noise. Identical frames are (0, 0, 1.0);
    offsets iterate by Manhattan distance with (0, 0) first, so ties resolve
    to no motion. Frames of different shape score (0, 0, 0.0) — no rigid
    translation explains a reshape.
    """
    if len(grid) != len(grid2) or any(
        len(r1) != len(r2) for r1, r2 in zip(grid, grid2)
    ):
        return 0, 0, 0.0
    changed = [
        (r, c)
        for r, row in enumerate(grid)
        for c, ch in enumerate(row)
        if grid2[r][c] != ch
    ]
    if not changed:
        return 0, 0, 1.0
    if len(changed) < 2:
        # One changed cell cannot distinguish translation from in-place
        # transformation — on a uniform background its symbol matches under
        # many offsets. The filter abstains rather than guess.
        return 0, 0, 0.0
    rows2, cols2 = len(grid2), len(grid2[0])
    offsets = sorted(
        (
            (dr, dc)
            for dr in range(-SHIFT_WINDOW, SHIFT_WINDOW + 1)
            for dc in range(-SHIFT_WINDOW, SHIFT_WINDOW + 1)
        ),
        key=lambda o: (abs(o[0]) + abs(o[1]), o[0], o[1]),
    )
    best_score = -1
    best_dr = best_dc = 0
    for dr, dc in offsets:
        score = 0
        for r, c in changed:
            r2, c2 = r + dr, c + dc
            if 0 <= r2 < rows2 and 0 <= c2 < cols2 and grid2[r2][c2] == grid[r][c]:
                score += 1
        if score > best_score:
            best_score = score
            best_dr, best_dc = dr, dc
    return best_dc, best_dr, best_score / len(changed)


def change_features(grid: list[list[str]], grid2: list[list[str]]) -> dict:
    """One frame change → the frame-diff family plus the translation fit.

    The one call site for both families: the bridge's per-transition
    features and the replay evidence's Measurement section
    (mechanical-correction-plan Phase D) must agree by construction, not
    by two implementations drifting apart."""
    features = _frame_diff(grid, grid2)
    dx, dy, fit = best_translation(grid, grid2)
    features["shift_dx"] = dx
    features["shift_dy"] = dy
    features["shift_fit"] = round(fit, 3)
    return features


def transition_features(
    action: str, obs: Obs, obs_next: Obs, held_keys: int, episode_t: int | None = None
) -> dict:
    """One transition → feature dict over FEATURE_KEYS (grids + host
    bookkeeping; `episode_t` rides along when the scan supplies it)."""
    grid, grid2 = obs.grid, obs_next.grid
    features: dict = {
        "action": action,
        "target_symbol": None,
        "moved": False,
        "target_symbol_after": None,
        "old_cell_symbol_after": None,
        "keys_delta": _count(grid2, "K") - _count(grid, "K"),
        "doors_delta": _count(grid2, "D") - _count(grid, "D"),
        "held_keys": held_keys,
        "episode_t": episode_t,
        # the env-recorded level/stage of the successor frame (honest
        # absence when the env records none — every current env)
        "level": (
            obs_next.info.get("level")
            if isinstance(obs_next.info, dict) else None
        ),
        "goal_reached": obs_next.goal_reached,
    }
    features.update(change_features(grid, grid2))
    agent = _find_agent(grid)
    # Internal marker (not in FEATURE_KEYS / the LLM vocabulary): whether the
    # agent-centric family is even measurable on this frame pair. A missing
    # 'A' cell makes all of them vacuous, which classify() treats as
    # no-evidence for patterns keyed on that family.
    features["_agent_present"] = agent is not None
    if agent is not None:
        delta = DIRECTIONS.get(action)
        features["old_cell_symbol_after"] = _cell(grid2, agent)
        agent_after = _find_agent(grid2)
        features["moved"] = agent_after is not None and agent_after != agent
        if delta is not None:
            target = (agent[0] + delta[0], agent[1] + delta[1])
            # Out of bounds behaves as a wall (the env's bump semantics).
            features["target_symbol"] = _cell(grid, target) or "#"
            features["target_symbol_after"] = _cell(grid2, target) or "#"
    return features


# -- patterns ---------------------------------------------------------------------


def _validate_side(side, name: str, required: bool) -> dict:
    if side is None:
        side = {}
    if not isinstance(side, dict):
        raise ValueError(f"pattern '{name}' must be an object")
    if required and not side:
        raise ValueError("pattern 'trigger' must be non-empty")
    for key, spec in side.items():
        if key not in FEATURE_KEYS:
            raise ValueError(f"unknown feature key {key!r}; expected one of {list(FEATURE_KEYS)}")
        if isinstance(spec, dict):
            if set(spec) != {"min"} or not isinstance(spec["min"], (int, float)):
                raise ValueError(f"comparator for {key!r} must be {{\"min\": <number>}}")
        elif not isinstance(spec, (str, int, float, bool)):
            raise ValueError(f"value for {key!r} must be a scalar or {{\"min\": n}}")
    return dict(side)


def validate_pattern(pattern) -> dict:
    """Normalize an (LLM-supplied) attribution pattern; ValueError when malformed."""
    if not isinstance(pattern, dict):
        raise ValueError("pattern must be a JSON object")
    return {
        "trigger": _validate_side(pattern.get("trigger"), "trigger", required=True),
        "outcome": _validate_side(pattern.get("outcome"), "outcome", required=False),
    }


def _matches(value, spec) -> bool:
    if isinstance(spec, dict):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and value >= spec["min"]
    return value == spec


def classify(pattern: dict, features: dict) -> str | None:
    """'confirm' | 'refute' | None for one transition against one pattern."""
    trigger, outcome = pattern["trigger"], pattern["outcome"]
    used_keys = set(trigger) | set(outcome)
    if not features.get("_agent_present", True) and used_keys & _AGENT_CENTRIC_KEYS:
        # Vacuous premises: the rule is stated over agent-centric features
        # but this world's frames have no agent, so the transition can
        # neither confirm nor refute it (a counterexample must share the
        # rule's premises). Seen live: 'moved' patterns on ARC frames.
        return None
    if not all(_matches(features.get(k), v) for k, v in trigger.items()):
        return None
    results = [_matches(features.get(k), v) for k, v in outcome.items()]
    if not results:
        return "confirm"  # trigger alone carries the rule's prediction
    return "confirm" if all(results) else "refute"


# -- history scans -------------------------------------------------------------------


def attribute_transitions_detailed(
    patterns: dict[str, dict], history: History
) -> tuple[dict[str, list[int]], dict[str, list[int]], dict[str, list[int]]]:
    """Full idempotent rescan: (rule id → [pos, neg], rule id → refuting t
    values ascending capped at FALSIFIED_BY_CAP, rule id → confirming t
    values ascending capped at CONFIRMED_BY_CAP — both halves of AWM's
    EVIDENCE list: the falsifier stays visible after the revision that
    answers it, and the confirmers make "rule confirms, engine diverges"
    (a code bug) readable beside "rule refutes" (a claim bug))."""
    counts = {rid: [0, 0] for rid in patterns}
    refutations: dict[str, list[int]] = {rid: [] for rid in patterns}
    confirmations: dict[str, list[int]] = {rid: [] for rid in patterns}
    for episode in history.episodes():
        held = 0
        episode_t = 0
        for tr in episode.transitions:
            features = transition_features(tr.action, tr.obs, tr.obs_next, held, episode_t)
            # Track held keys from the ground deltas (pickup +1, door-open −1).
            held += max(0, -features["keys_delta"])
            held = max(0, held + min(0, features["doors_delta"]))
            episode_t += 1
            for rid, pattern in patterns.items():
                verdict = classify(pattern, features)
                if verdict == "confirm":
                    counts[rid][0] += 1
                    if len(confirmations[rid]) < CONFIRMED_BY_CAP:
                        confirmations[rid].append(tr.t)
                elif verdict == "refute":
                    counts[rid][1] += 1
                    if len(refutations[rid]) < FALSIFIED_BY_CAP:
                        refutations[rid].append(tr.t)
    return counts, refutations, confirmations


def attribute_transitions(patterns: dict[str, dict], history: History) -> dict[str, list[int]]:
    """Full idempotent rescan: rule id → [pos, neg] over the whole record."""
    return attribute_transitions_detailed(patterns, history)[0]


def update_from_history(store: RuleStore, history: History, iteration: int) -> dict:
    """Apply the rescan to a rule store; returns per-rule deltas for the
    ledger (rules whose counts changed, with their revised truth)."""
    patterns = {}
    for rule in store.rules:
        if rule.pattern:
            try:
                patterns[rule.id] = validate_pattern(rule.pattern)
            except ValueError:
                continue  # an unusable pattern earns no evidence, never a crash
    counts, refutations, confirmations = (
        attribute_transitions_detailed(patterns, history) if patterns else ({}, {}, {})
    )
    updates: dict[str, dict] = {}
    for rule in store.rules:
        pos, neg = counts.get(rule.id, (rule.pos, rule.neg))
        refs = refutations.get(rule.id, [])
        cons = confirmations.get(rule.id, [])
        counts_changed = (pos, neg) != (rule.pos, rule.neg)
        refs_changed = refs != rule.falsified_by
        cons_changed = cons != rule.confirmed_by
        fate = FATE_REFUTED if (pos == 0 and neg > 0) else FATE_ACTIVE
        fate_changed = rule.fate != fate
        if counts_changed or refs_changed or cons_changed or fate_changed:
            if counts_changed:
                rule.pos, rule.neg = pos, neg
                rule.updated_iteration = iteration
            rule.falsified_by = list(refs)
            rule.confirmed_by = list(cons)
            rule.fate = fate
            tv = rule.truth
            updates[rule.id] = {
                "pos": rule.pos,
                "neg": rule.neg,
                "truth": [round(tv.f, 3), round(tv.c, 3)],
                "falsified_by": list(rule.falsified_by),
                "confirmed_by": list(rule.confirmed_by),
                "fate": rule.fate,
            }
    if updates:
        store.save()
    return updates
