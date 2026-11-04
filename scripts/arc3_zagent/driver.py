"""The episode driver: one ARC-AGI-3 game played through the zAgent mind.

Loop per step (the observe → think → act posture, no LLM in the loop):

1. FEED     the previous transition's percept events (`input` with the
            event-loop cadence — 6 cycles/event + 20 tail). Every reply
            drains the act queue into the pending pool.
2. PROPOSE  take the first valid act from the pool (the mind's proposal);
            decline the rest (`performed=false`, the calibrated decline).
3. FALLBACK when the mind proposes nothing, a seeded-random action keeps the
            evidence stream dense — logged as `proposed_by: "random"`.
4. PERFORM  `env.step(action)`; land the motor report (act-resolution for
            proposed acts, `performed_event` for fallbacks — one stream,
            never both).
5. RECORD   one JSONL row per step; on the episode boundary, land the
            boundary events and stop.

Goals are tasked ONCE per registered op (`<(*,{SELF}) --> ^op>! %1.0;0.9%`),
at registration — the decision route fires each goal at most once, so the
mind's proposal burst is bounded and later proposals are evidence-driven
(goal revisions), not harness-paced. That restraint is deliberate: re-tasking
every step would make the proposal stream a shadow of the harness's choice
rather than a measurement of the mind's.
"""

from __future__ import annotations

import json
import random
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .percepts import (
    action_of_op,
    boundary_events,
    budget_event,
    change_probe,
    op_name,
    performed_event,
    transition,
)

EVENT_CYCLE_PER_EVENT = 6
EVENT_CYCLE_TAIL = 20
PROBE_CYCLES = 50


@dataclass
class StepRow:
    step: int
    action: str
    op: str
    proposed_by: str  # "mind" | "random"
    acts_proposed: int
    acts_declined: int
    events_emitted: int
    changed_cells: int | None = None
    appeared: str | None = None
    disappeared: str | None = None
    shift: str | None = None
    arc3_state: str | None = None
    levels_completed: int | None = None
    goal_reached: bool = False
    # Step provenance (the sim's worldRows-hit vs stepRules-inference,
    # surfaced by LeanSimTransport.last_via): None on venues without it
    # (the live API). A level-up whose via is not "worldRows" is the
    # silent-divergence event class — logged, never papered over.
    via: str | None = None
    refused_lines: list[str] = field(default_factory=list)
    mind_time: int | None = None
    mind_concepts: int | None = None


@dataclass
class EpisodeResult:
    summary: dict
    log_rows: list[StepRow]


class Arc3Driver:
    def __init__(
        self,
        env,
        mind,
        *,
        seed: int = 42,
        log=print,
        retask_per_episode: bool = False,
        action_delay: float = 0.0,
        record_path: Path | None = None,
        objective=None,
        extra_percept=None,
        plan_lock: bool = False,
    ):
        self.env = env
        self.record_path = record_path
        self.plan_lock = plan_lock
        # The generated objective (adoption/objective.py): its drive
        # effects are tasked as outcome goals at registration, its
        # posture rides warm_start. None preserves the legacy posture.
        self.objective = objective
        # venue-side frame-derived percepts (the ls20 sim's [hud_match]
        # discriminator): called as fn(pre_grid, grid, info) -> lines;
        # render-side only, inert when None
        self.extra_percept = extra_percept
        # R0.1 (reflection corpus): mirror every learning command to
        # events.jsonl. Off by default — without it, `mind` is untouched.
        if record_path is not None:
            from .record import RecordingMind

            self.mind = RecordingMind(mind, record_path)
        else:
            self.mind = mind
        self.rng = random.Random(seed)
        self.log = log
        self.pending: deque[dict] = deque()
        self.registered: set[str] = set()
        self.rows: list[StepRow] = []
        self.total_refused: list[dict] = []
        self.warm: bool = False
        self.episodes = 0
        # Mission posture (grind runs): re-task every registered op's goal at
        # the start of each episode, so the mind gets a proposal burst per
        # attempt instead of once per interface. Logged in the summary.
        self.retask_per_episode = retask_per_episode
        # Politeness: seconds to sleep after each env step. The server
        # 401-throttled the key after ~6,700 actions at ~520/min; a paced
        # walk (~120/min at 0.5) is the resume posture.
        self.action_delay = action_delay
        # The route bridge (mind_route.author_bridge's line, set by the
        # loop at a sighting boundary): a budgeted &/ sequence goal the
        # drive pulse re-tasks every PULSE_EVERY steps — each task is a
        # fresh stamp, i.e. a fresh observe-lottery ticket (measured
        # 2026-11-03: one tasking takes ~370 cycles to surface under a
        # mining-flooded overall bag; the pulse cadence buys ~26-cycle
        # retries), and each acceptance re-runs the decision route's
        # sequence decomposition (the mind's own first-component
        # execution — proposed_by stays "mind", never the body's).
        self.route_bridge: str | None = None

    def set_record_context(self, **kw) -> None:
        """Stamp the recorder's step context (no-op when not recording)."""
        if self.record_path is not None:
            self.mind.context.update(kw)

    def set_event_window(self, capacity: int) -> dict:
        """The PTCG venue's event-buffer knob, exposed for the goal-
        sighting posture: the reference window is 3 (Config's
        EVENT_BUFFER_CAPACITY), so a sighted route's tail never
        co-buffers with its boundary event and nothing goal-linked can
        mine (the drive_unbridged F4 alarm's structural cause). Widened
        only for sighting attempts — the mined rules survive a shrink
        (they are memory residents, not buffer rows)."""
        reply = self._command({"op": "configure",
                               "event_buffer_capacity": capacity})
        self.log(f"event window: {capacity} "
                 f"({(reply.get('lines') or [''])[0]})")
        return reply

    def set_route_bridge(self, narsese: str | None) -> None:
        """Install (or clear) the sequence-goal bridge the pulse
        re-tasks. Authored once per sighting boundary by the loop."""
        self.route_bridge = narsese

    # -- mind plumbing ---------------------------------------------------------------

    def _command(self, cmd: dict) -> dict:
        reply = self.mind.call(cmd)
        self.pending.extend(reply.get("acts") or [])
        self.total_refused.extend(reply.get("refused") or [])
        return reply

    class _InputShim:
        """The weave's `mind.input(line)` form over the driver's
        `_command` path — refusals land in the driver's ledger."""

        def __init__(self, command):
            self._command = command

        def input(self, narsese, **kw):
            return self._command({"op": "input", "narsese": narsese})

    def _register_ops(self, actions: list[str]) -> None:
        fresh = []
        for action in actions:
            op = op_name(action)
            if op in self.registered:
                continue
            self._command({"op": "register_op", "name": op})
            # The goal is tasked once per op: the desire the decision route
            # can act on. 0.9 is the zagent-minimal reference desire.
            self._command({"op": "input", "narsese": f"<(*,{{SELF}}) --> ^{op}>! %1.0;0.9%"})
            self.registered.add(op)
            fresh.append(op)
        if fresh:
            self.log(f"registered ops: {', '.join(fresh)}")
        if self.objective is not None:
            from adoption.objective_weave import task_drive_goals
            tasked = task_drive_goals(
                Arc3Driver._InputShim(self._command),
                self.objective, identity=self)
            if tasked:  # deduped per (identity, drive) — log once, not per step
                self.log(f"drive tasked from objective: "
                         f"{list(self.objective.drive_effects)}")

    def _retask(self) -> None:
        """Fresh desires for every registered op (grind posture)."""
        narsese = "\n".join(
            f"<(*,{{SELF}}) --> ^{op}>! %1.0;0.9%" for op in sorted(self.registered)
        )
        reply = self.mind.input(narsese)
        self.pending.extend(reply.get("acts") or [])
        self.total_refused.extend(reply.get("refused") or [])

    def _take_act(self, valid_ops: set[str]) -> tuple[dict | None, int]:
        """First valid pending act; every other pending act is declined (the
        fire-and-forget doctrine: unperformed is landed as a low-confidence
        non-occurrence, never silently dropped). Under plan_lock every
        pending act is declined — the route owns the actions; the mind's
        proposals are landed as non-occurrences, and its perception of the
        episode is untouched."""
        chosen: dict | None = None
        declined = 0
        while self.pending:
            act = self.pending.popleft()
            if chosen is None and act["op"] in valid_ops and not self.plan_lock:
                chosen = act
                continue
            self._command({"op": "observe", "act_id": act["id"], "performed": False})
            declined += 1
        return chosen, declined

    # -- the loop ----------------------------------------------------------------------

    def warm_start(self, snapshot_path: Path) -> dict:
        """Restore a previous episode's mind snapshot (G-Z2's rebind: the
        bridge registry comes back empty in a fresh process, so the ops are
        re-registered and their goals re-tasked by the next `run()` —
        restored *beliefs*, fresh *desires*)."""
        data = Path(snapshot_path).read_bytes()
        reply = self.mind.restore(data)
        self.warm = True
        refused = reply.get("refused") or []
        self.log(f"warm start: {snapshot_path} → {reply.get('lines', [])} refused={len(refused)}")
        if refused:
            raise RuntimeError(f"warm start refused: {refused}")
        if self.objective is not None and self.objective.concept_headroom:
            from adoption.objective_weave import apply_restore_posture
            self.log(f"restore opener: {apply_restore_posture(self.mind, self.objective)}")
        return reply

    def run(self, *, max_actions: int, workspace: Path | None = None,
            plan: list[str] | None = None,
            boundary_stop: str = "any") -> EpisodeResult:
        """`plan` (the frontier campaign): when the mind proposes
        nothing, the fallback consumes the plan's op for this step
        instead of sampling — logged as `proposed_by: "plan"` so
        attribution never confuses the campaign's routes with either
        the mind or chance. A plan op outside the live interface falls
        through to the seeded sample (honest, logged).

        `boundary_stop` — "any" (default: the episode ends at the first
        level-up, the minimize loop's metric unit) or "terminal" (only
        WIN/GAME_OVER end it: the deep-sighting posture, where one
        injected episode can carry the mind through ALL of a game's
        level-ups instead of only the first)."""
        self.episodes += 1
        row_start = len(self.rows)  # rows accumulate across episodes; the
        # per-episode report is the delta (the minimize loop's metric)
        obs = self.env.reset()
        info = obs.info or {}
        self.log(
            f"reset: game={self.env.game_id} state={info.get('arc3_state')} "
            f"actions={self.env.actions} "
            f"levels={info.get('levels_completed')}/{info.get('win_levels')}"
        )
        self._register_ops(self.env.actions)
        if self.retask_per_episode and self.episodes > 1:
            self._retask()

        boundary: str | None = "reset-boundary" if obs.goal_reached else None

        for step in range(max_actions):
            if boundary:
                break
            action, obs, step_boundary = self._step_once(obs, plan, step)
            if step_boundary is not None:
                terminal = step_boundary in ("WIN", "GAME_OVER")
                if boundary_stop == "any" or terminal:
                    boundary = step_boundary
                    break
                # level-up with terminal-only stop: keep playing — the
                # sighting stream continues into the next level
                self.log(f"level-up passed through (deep sighting {step_boundary})")

        if boundary is None:
            boundary = "action-cap"
            self.log(f"no boundary within {max_actions} actions; stopping")

        summary = self._summarize(boundary, row_start)
        if workspace is not None:
            self._write(workspace, summary, row_start)
        return EpisodeResult(summary=summary,
                             log_rows=self.rows[row_start:])

    def _step_once(self, obs, plan, step: int):
        """One action of the episode loop (extracted verbatim from run();
        the Kaggle notebook adapter calls this directly — one
        choose_action, one step). Returns (action, obs, boundary) where
        boundary is None unless the step closed the episode."""
        self.set_record_context(step=step)

        # 1-3. PROPOSE / FALLBACK (interfaces can grow mid-game: re-register).
        self._register_ops(self.env.actions)
        valid_ops = {op_name(a) for a in self.env.actions}
        chosen, declined = self._take_act(valid_ops)
        if chosen is not None:
            action = action_of_op(chosen["op"])
            proposed_by = "mind"
        elif plan is not None and step < len(plan) \
                and plan[step] in valid_ops:
            action = action_of_op(plan[step])
            proposed_by = "plan"
        else:
            action = self.rng.choice(self.env.actions)
            proposed_by = "random"

        # 4. PERFORM + OBSERVE the consequence (the embodied order: the
        # act first, then its percept batch, chewed at the event-loop
        # cadence — the next step's proposals see everything).
        op = op_name(action)
        pre_grid = obs.grid
        try:
            obs = self.env.step(action)
        except ValueError:
            # Interface shrank between the check and the step (rare):
            # the act is stale — decline it honestly, re-pick randomly.
            if chosen is not None:
                self._command(
                    {"op": "observe", "act_id": chosen["id"], "performed": False}
                )
                chosen = None
            action = self.rng.choice(self.env.actions)
            proposed_by = "random-retry"
            op = op_name(action)
            obs = self.env.step(action)
        if chosen is not None:
            self._command(
                {"op": "observe", "act_id": chosen["id"], "performed": True}
            )
        else:
            self.mind.input(performed_event(op))
        info = obs.info or {}
        via = getattr(getattr(self.env, "transport", None), "last_via", None)
        report = transition(op, pre_grid, obs.grid)
        if self.action_delay:
            time.sleep(self.action_delay)
        batch = list(report.events)
        band_emitted = False
        pulse_lines: list[str] = []
        if self.objective is not None:
            # The win-enabling minimum (the three-venue lesson), arc3
            # edition — BOTH render/task-side, both inert without a
            # generated objective (the legacy posture is byte-identical):
            # (a) the approach/progress signal — the budget band against
            #     the objective's pre-registered ceiling;
            # (b) the mid-episode drive pulse — the decision route fires
            #     each goal at most once, so the once-tasked drive never
            #     re-scans; the pulse re-alarms it on the measured
            #     gridworld cadence (identical lines merge into the
            #     resident desire; the pulse exists to re-scan).
            batch.append(budget_event(step, self.objective.action_ceiling))
            band_emitted = True
            if step and step % 6 == 0:
                pulse_lines = [f"<{{SELF}} --> [{d}]>! %1.0;0.9%"
                               for d in self.objective.drive_effects]
            if self.route_bridge is not None and step and step % 6 == 0:
                # the bridge rides the same pulse cadence: a fresh stamp
                # is a fresh lottery ticket, and its acceptance re-runs
                # the decision route's sequence decomposition
                pulse_lines.append(self.route_bridge)
        # Episode-boundary events ride the SAME batch as the step's
        # percepts (the engine mines within-batch co-occurrence — the
        # separate boundary input starved every drive chain: measured
        # 2026-11-03, zero [level_up] implications after 4 sightings).
        # Same events, same stream — render-side only, no beliefs.
        if obs.goal_reached:
            leveled = bool(self.rows) and (
                (info.get("levels_completed") or 0)
                > (self.rows[-1].levels_completed or 0))
            batch.extend(boundary_events(info.get("arc3_state") or "",
                                         leveled))
        # venue-side frame-derived percepts (the ls20 sim's [hud_match]
        # discriminator): render-side only, inert when None
        if self.extra_percept is not None:
            batch.extend(self.extra_percept(pre_grid, obs.grid, info) or [])
        reply = self.mind.input(
            "\n".join(batch),
            per_event=EVENT_CYCLE_PER_EVENT,
            tail=EVENT_CYCLE_TAIL,
        )
        for line in pulse_lines:
            pulse_reply = self.mind.input(line)
            self.pending.extend(pulse_reply.get("acts") or [])
            self.total_refused.extend(pulse_reply.get("refused") or [])
        self.pending.extend(reply.get("acts") or [])
        self.total_refused.extend(reply.get("refused") or [])

        # 5. RECORD.
        state = self.mind.state()
        row = StepRow(
            step=step,
            action=action,
            op=op,
            proposed_by=proposed_by,
            acts_proposed=1 if chosen is not None else 0,
            acts_declined=declined,
            events_emitted=len(batch),
            changed_cells=report.changed_cells,
            appeared=report.appeared,
            disappeared=report.disappeared,
            shift=report.shift,
            arc3_state=info.get("arc3_state"),
            levels_completed=info.get("levels_completed"),
            goal_reached=obs.goal_reached,
            via=via,
            mind_time=state.get("state", {}).get("time"),
            mind_concepts=state.get("state", {}).get("concepts"),
        )
        self.rows.append(row)
        self.log(
            f"step {step:3d}: {action:16s} by={proposed_by:13s} "
            f"declined={declined} events={row.events_emitted} "
            f"cells={row.changed_cells} state={row.arc3_state}"
        )

        boundary = None
        if obs.goal_reached:
            # leveled was computed pre-input (boundary events rode the
            # batch); the boundary only classifies + logs here now.
            leveled = bool(self.rows) and (
                (info.get("levels_completed") or 0)
                > (self.rows[-1].levels_completed or 0))
            raw = info.get("arc3_state")
            boundary = (
                raw if raw in ("WIN", "GAME_OVER") else "level-up"
            )
            self.log(f"boundary: {boundary} (leveled={leveled}) after {step + 1} action(s)")
        return action, obs, boundary

    # -- post-episode -----------------------------------------------------------------

    def probe_change_beliefs(self) -> dict:
        """Ask the mind, per op, whether the frame changes under it. The
        answers are whatever the belief tables hold — empty is an honest
        answer and the analysis compares them to the observed rates."""
        ops = sorted(self.registered)
        probe = {}
        for op, question in change_probe(ops):
            reply = self.mind.ask(question, cycles=PROBE_CYCLES)
            answers = [ln for ln in reply.get("lines", []) if ln.startswith(("answer:", "quest-answer:"))]
            probe[op] = answers
        return probe

    def _summarize(self, boundary: str, row_start: int = 0) -> dict:
        by_op: dict[str, dict] = {}
        for row in self.rows[row_start:]:
            slot = by_op.setdefault(row.op, {"steps": 0, "by_mind": 0, "changed": 0})
            slot["steps"] += 1
            slot["by_mind"] += row.proposed_by == "mind"
            slot["changed"] += bool(row.changed_cells)
        mind_rows = sum(r.proposed_by.startswith("mind")
                        for r in self.rows[row_start:])
        return {
            "game_id": self.env.game_id,
            "warm_start": self.warm,
            "episode": self.episodes,
            "retask_per_episode": self.retask_per_episode,
            "boundary": boundary,
            "steps": len(self.rows) - row_start,
            "levels_completed": self.rows[-1].levels_completed if self.rows else 0,
            "mind_share": (mind_rows / (len(self.rows) - row_start)
                           if len(self.rows) > row_start else 0.0),
            "acts_declined_total": sum(r.acts_declined
                                       for r in self.rows[row_start:]),
            "refused_total": len(self.total_refused),
            "per_op": by_op,
            "change_probe": self.probe_change_beliefs(),
            "final_state": self.mind.state().get("state"),
        }

    def _write(self, workspace: Path, summary: dict,
               row_start: int = 0) -> None:
        workspace = Path(workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        with (workspace / "run_log.jsonl").open("w", encoding="utf-8") as f:
            for row in self.rows[row_start:]:  # this episode's rows only
                f.write(json.dumps(asdict(row)) + "\n")
        (workspace / "episode_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        snap = self.mind.snapshot()
        if snap.get("snapshot") is not None:
            (workspace / "mind_snapshot.bin").write_bytes(bytes(snap["snapshot"]))
        self.log(f"wrote {workspace}/run_log.jsonl, episode_summary.json, mind_snapshot.bin")
