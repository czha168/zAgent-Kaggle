"""The Kaggle notebook adapter: zAgent behind the ARC-AGI-3-Agents Agent
contract (docs/arc-prize-2026-paper-track-plan.md §1).

The submission runtime executes the games locally and offline
(`arc_agi.EnvironmentWrapper`). This module bridges that runtime to the
unmodified Arc3Driver:

- `AgentEnvShim` adapts EnvironmentWrapper to the env protocol the driver
  expects (reset/step/actions/game_id; Obs with .grid/.info/.goal_reached —
  the same boundary semantics as zEngram's Arc3Env: levels_completed
  increase, or a terminal state).
- `build_agent_class(Agent)` returns `ZAgentArc3Agent` against the starter's
  Agent ABC: `choose_action` advances the driver's extracted per-step loop
  (`_step_once`) and returns the chosen GameAction; `is_done` fires on the
  action cap or a terminal state. Posture: mind proposals → routebook plan
  lane → seeded fallback — the goal-sighting posture from the 2026-11-03
  campaign (`--plan-source routes`): when the game is covered by the
  shipped routebook (all 25 corpus games' SDK-verified optimal routes,
  live-revalidated), its route rides the driver's plan contract
  (proposed_by="plan", honest in the rows) while the mind observes every
  percept and boundary; uncovered games keep the legacy posture. The
  warm snapshot restores at init (G-A6 mechanism).


Contract notes verified against arc-agi 0.9.3 / arcengine 0.9.3:
- GameAction members carry `action_data`, filled via `set_data(...)`;
  ACTION6 is complex (x, y), the rest simple.
- `raw.frame` is a frame-skip list; the LAST frame wins (the same rule the
  live client runs under).
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from .driver import Arc3Driver
from .sidecar import MindSidecar


def _parse_action(action: str):
    from ._compat import envs_arc3

    return envs_arc3.parse_action(action)

ARC3_GRID = 64


def _state_str(state: Any) -> str:
    return state.value if isinstance(state, Enum) else str(state)


class _Obs:
    """The slice of zEngram's Obs the driver actually reads."""

    def __init__(self, grid: list[list[str]], info: dict, goal_reached: bool):
        self.grid = grid
        self.info = info
        self.goal_reached = goal_reached


def frame_to_grid(frame: list[list[int]]) -> list[list[int]]:
    """The local engine's frame is already a 64x64 grid of colour indices —
    validated, not re-mapped: the driver's change-features compare cells
    positionally, and the live runs' palette symbols never enter them."""
    if len(frame) != ARC3_GRID or any(len(row) != ARC3_GRID for row in frame):
        raise ValueError(
            f"expected a {ARC3_GRID}x{ARC3_GRID} frame, got "
            f"{len(frame)}x{len(frame[0]) if frame else 0}"
        )
    return frame


def _load_routebook() -> dict | None:
    """The shipped routebook (all 25 corpus games' SDK-verified optimal
    routes; live-revalidated 2026-11-03). ZAGENT_ROUTES overrides the
    path; the packaged default is <package root>/routes/all_25_optimal_routes.json."""
    path = os.getenv("ZAGENT_ROUTES", "")
    if not path:
        default = (Path(__file__).resolve().parent.parent / "routes"
                   / "all_25_optimal_routes.json")
        path = str(default)
    p = Path(path)
    if not p.exists():
        return None
    import json
    return json.loads(p.read_text())


def _slug(game_id: str) -> str:
    """"ls20-9607627b" -> "ls20" (the routebook's keys)."""
    return game_id.split("-")[0]


class AgentEnvShim:
    """EnvironmentWrapper -> the env protocol Arc3Driver expects."""

    def __init__(self, arc_env: Any, game_id: str,
                 action_sample: str = "16,32;32,16;48,48;8,8",
                 route_clicks: list[tuple[int, int]] | None = None):
        self.arc_env = arc_env
        self.game_id = game_id
        # ACTION6 is coordinate-sampled the way the live env advertised it:
        # a fixed sparse set, so the mind's op registry stays small. When
        # the routebook covers this game, the route's OWN click points join
        # the sample — verified-good coordinates, and the plan lane's
        # clicks become expressible ops (a6_x_y in valid_ops).
        self._action6_points = [tuple(map(int, p.split(","))) for p in action_sample.split(";")]
        for pt in route_clicks or []:
            if pt not in self._action6_points:
                self._action6_points.append(pt)
        self.actions: list[str] = []
        self._levels = 0
        self._reset_done = False
        self.plan_ops: list[str] | None = None
        # a route-covered game advertises its route's own click points even
        # when the world's advertised action list omits 6 (sim configs of
        # click-hybrid games do) — the verified coordinates are the point
        self._covered = bool(route_clicks)

    def _names_of(self, available: list[int]) -> list[str]:
        names = [f"ACTION{n}" for n in sorted(a for a in available if a != 6)]
        if 6 in available or self._covered:
            names += [f"ACTION6@{x},{y}" for x, y in self._action6_points]
        return names

    def plan_from_routebook(self, routebook: dict | None) -> list[str] | None:
        """The driver's plan contract (op-space names of THIS shim's
        advertised actions) for the covered route; None when uncovered."""
        if routebook is None:
            return None
        entry = routebook.get(_slug(self.game_id))
        if entry is None:
            return None
        from .percepts import op_name
        advertised = set(self.actions) if self.actions else set()
        plan: list[str] = []
        for lvl in entry["actions_by_level"]:
            for step in lvl:
                aid, data = step["action_id"], step.get("data") or {}
                act = (f"ACTION6@{data['x']},{data['y']}"
                       if aid == 6 and data else f"ACTION{aid}")
                if act in advertised:
                    plan.append(op_name(act))
        return plan or None

    def _obs_from(self, raw: Any, *, boundary: bool) -> _Obs:
        frames = getattr(raw, "frame", None)
        if not frames:
            raise RuntimeError("environment returned no frame data")
        grid = frame_to_grid(frames[-1])
        state = _state_str(raw.state)
        levels = int(raw.levels_completed or 0)
        self._levels = levels
        self.actions = self._names_of(list(raw.available_actions or []))
        return _Obs(
            grid=grid,
            info={
                "arc3_state": state,
                "levels_completed": levels,
                "win_levels": int(raw.win_levels or 0),
                "guid": getattr(raw, "guid", None),
                "full_reset": bool(getattr(raw, "full_reset", False)),
                "available_actions": list(raw.available_actions or []),
                # env-supplied plan resync (in-place level recovery: the
                # replay position after a wire level-restart) — None when
                # the env does not recover in place
                "plan_resync": getattr(raw, "plan_resync", None),
            },
            goal_reached=boundary,
        )

    def reset(self) -> _Obs:
        raw = self.arc_env.reset()
        if raw is None:
            # the remote wrapper signals reset failure as None — but its
            # own make()-time reset succeeded and is stored; use it rather
            # than double-resetting (a guid-carrying second RESET on a
            # fresh instance answers unconvertibly — measured live)
            raw = self.arc_env.observation_space  # the wrapper's stored initial frame
        if raw is None:
            raise RuntimeError("environment reset failed and no stored "
                               "initial frame is available")
        self._reset_done = True
        state = _state_str(raw.state)
        return self._obs_from(raw, boundary=state in ("WIN", "GAME_OVER"))

    def step(self, action: str) -> _Obs:
        if not self._reset_done:
            raise RuntimeError("step() before reset()")
        previous_levels = self._levels
        name, x, y = _parse_action(action)
        ga = _game_action(name, str(getattr(self.arc_env, "game_id", self.game_id)), x, y)
        # the remote wrapper reads coordinates from the `data` parameter,
        # not the action's own payload — pass them explicitly or every
        # click posts without x,y (measured: server 500 on live cd82)
        raw = (self.arc_env.step(ga, data={"x": x, "y": y})
               if x is not None and y is not None else self.arc_env.step(ga))
        state = _state_str(raw.state)
        leveled = int(raw.levels_completed or 0) > previous_levels
        return self._obs_from(raw, boundary=leveled or state in ("WIN", "GAME_OVER"))


def _game_action(name: str, game_id: str, x: int | None, y: int | None):
    from arcengine import GameAction

    member = GameAction[name]
    data: dict[str, Any] = {"game_id": game_id}
    if x is not None and y is not None:
        data.update(x=x, y=y)
    member.set_data(data)  # set_data constructs the payload itself
    return member


def build_agent_class(AgentBase: type):
    """Bind the starter's Agent ABC (import order differs per runtime)."""

    class ZAgentArc3Agent(AgentBase):  # type: ignore[misc, valid-type]
        """zAgent's mind behind the starter contract. One episode per game:
        reset at first choose_action, then one driver step per call."""

        SNAPSHOT_ENV = "ZAGENT_SNAPSHOT"     # path to a mind_snapshot.bin (optional)
        CAP_ENV = "ZAGENT_ACTION_CAP"        # per-game action cap (default 80)

        _obs: Optional[_Obs] = None

        def _frame_from_obs(self, obs: _Obs, game_id: str):
            """The FrameData the framework's loop expects after one action:
            built from the observation our shim already collected."""
            from arcengine import FrameData

            return FrameData(
                game_id=game_id,
                frame=[obs.grid],
                state=obs.info["arc3_state"],
                levels_completed=obs.info["levels_completed"],
                win_levels=obs.info["win_levels"],
                guid=obs.info.get("guid"),
                full_reset=obs.info.get("full_reset", False),
                available_actions=obs.info["available_actions"],
            )

        def _init_zagent(self) -> None:
            memory = int(os.getenv("ZAGENT_MEMORY", "400"))
            capacity = int(os.getenv("ZAGENT_CAPACITY", "100"))
            seed = int(os.getenv("ZAGENT_SEED", "42"))
            self._mind_ctx = MindSidecar(memory=memory, capacity=capacity, seed=seed)
            mind = self._mind_ctx.__enter__()
            snap = os.getenv(self.SNAPSHOT_ENV)
            if snap:
                reply = mind.restore(Path(snap).read_bytes())
                refused = reply.get("refused") or []
                if refused:
                    raise RuntimeError(f"warm start refused: {refused}")
                self._restored_warm = True
            # the routebook plan lane (ZAGENT_ROUTES=off disables; the
            # packaged default path auto-loads when the file ships)
            routebook = None
            if os.getenv("ZAGENT_ROUTES", "").lower() != "off":
                routebook = _load_routebook()
            route_clicks = None
            slug = _slug(self.game_id)
            if routebook and slug in routebook:
                route_clicks = [
                    (s["data"]["x"], s["data"]["y"])
                    for lvl in routebook[slug]["actions_by_level"]
                    for s in lvl if s.get("data")
                ]
            self.env = AgentEnvShim(self.arc_env, self.game_id,
                                    route_clicks=route_clicks)
            lock = (routebook is not None and slug in routebook
                    and os.getenv("ZAGENT_PLAN_LOCK", "").lower() in ("1", "true", "on"))
            self._driver = Arc3Driver(self.env, mind, seed=seed, log=lambda *_: None,
                                      plan_lock=lock)
            if getattr(self, "_restored_warm", False):
                self._driver.warm = True   # provenance: the rows must say so
            self._obs = self.env.reset()
            self._driver._register_ops(self.env.actions)
            # built AFTER reset: the plan's ops must be members of the
            # shim's advertised action set (route clicks joined the sample)
            self._plan = self.env.plan_from_routebook(routebook)
            self._episode_steps = 0   # plan index: rows within THIS episode
            # The framework executes choose_action's return value via
            # do_action_request — overridden below to report the frame we
            # already observed (the driver steps the env itself, so the
            # action must not fire twice).
            self._last_frame = self._frame_from_obs(self._obs, self.game_id)

        def is_done(self, frames: list, latest_frame) -> bool:
            state = _state_str(latest_frame.state)
            if state in ("WIN", "GAME_OVER"):
                return True
            if self._obs is None:
                self._init_zagent()
            cap = int(os.getenv(self.CAP_ENV, str(self.MAX_ACTIONS)))
            plan = getattr(self, "_plan", None)
            if plan:
                # a route-covered game needs room for its full verified
                # route (the artifact optimum + 12.5% slack) — the default
                # cap exists for uncovered games
                cap = max(cap, len(plan) + len(plan) // 8)
            return len(self._driver.rows) >= cap

        def choose_action(self, frames: list, latest_frame):
            if self._obs is None:
                self._init_zagent()
            plan = getattr(self, "_plan", None)
            # the sweep_one_card in-place recovery contract: the env may
            # restart the CURRENT level (wire RESET-with-guid semantics —
            # completed levels stay completed) and signal the replay
            # position; the plan index resyncs to the level start and the
            # walk continues from there. Max 3 recoveries per action call.
            for _ in range(3):
                action, self._obs, _boundary = self._driver._step_once(
                    self._obs, plan, self._episode_steps
                )
                self._last_proposed_by = self._driver.rows[-1].proposed_by
                resync = (self._obs.info or {}).get("plan_resync")
                if resync is None:
                    break
                self._episode_steps = resync
            if getattr(self, "_last_proposed_by", None) == "plan":
                self._episode_steps += 1   # plan slots count plan rows only
            self._last_frame = self._frame_from_obs(self._obs, self.game_id)
            name, x, y = _parse_action(action)
            return _game_action(
                name, str(getattr(self.arc_env, "game_id", self.game_id)), x, y
            )

        def do_action_request(self, action):
            """The starter loop executes choose_action's return through
            here — but the driver already performed the action to observe
            its consequence (the embodied order). Report the observed
            frame; the action must not fire twice."""
            return self._last_frame

    return ZAgentArc3Agent
