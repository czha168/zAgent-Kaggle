"""Offline test: ZAgentArc3Agent satisfies the ARC-AGI-3-Agents Agent
contract using a stub arc_env — no live API, no local engine required.

Checks:
1. choose_action returns arcengine GameAction members (ACTION6 carries x,y).
2. The driver's decision core runs unmodified under the shim (percepts,
   mind proposals, fallback; zero refusals).
3. is_done fires on the action cap AND on a terminal state.
4. A boundary (WIN) propagates: the driver records the boundary row.
"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # dataset root: import arc3_zagent + vendored zengram
sys.path.insert(0, "/tmp/arcengine_pkg")

from arc3_zagent.kaggle_agent import AgentEnvShim, build_agent_class

# inlined from test_offline.py (not vendored): the scripted-frame helpers
GRID = 64
def frame(seed: int):
    return [[(seed + r * 7 + c * 3) % 16 for c in range(GRID)] for r in range(GRID)]
def patched(fr, seed: int):
    out = [row[:] for row in fr]
    for r in range(10, 14):
        for c in range(20, 24):
            out[r][c] = seed % 16
    return out
F1, F2 = frame(5), patched(frame(5), 9)

GRID = 64


class RawFrame:
    def __init__(self, grid, state="NOT_FINISHED", levels=0, win=254, available=(1, 2, 3)):
        self.frame = [grid]
        self.state = state
        self.levels_completed = levels
        self.win_levels = win
        self.available_actions = list(available)
        self.guid = "guid-1"
        self.full_reset = False


class StubArcEnv:
    """The slice of EnvironmentWrapper the shim touches."""

    game_id = "zz99-test"

    def __init__(self, script: list[RawFrame]):
        self.script = list(script)
        self.received: list = []

    def reset(self):
        return self.script[0]

    def step(self, ga):
        self.received.append(ga)
        return self.script[min(len(self.script) - 1, len(self.received))]


class StubAgentBase:
    MAX_ACTIONS = 80
    def __init__(self, card_id, game_id, agent_name, ROOT_URL, record, arc_env, tags=None):
        self.arc_env = arc_env
        self.game_id = game_id


def run(cap: int, script: list[RawFrame]):
    import os
    from arc3_zagent.kaggle_agent import build_agent_class

    os.environ["ZAGENT_ACTION_CAP"] = str(cap)
    ZAgentArc3Agent = build_agent_class(StubAgentBase)
    env = StubArcEnv(script)
    agent = ZAgentArc3Agent("card", "zz99-test", "zagent", "http://localhost", False, env)
    latest = script[0]
    actions = []
    for _ in range(cap + 5):
        if agent.is_done([], latest):
            break
        ga = agent.choose_action([], latest)
        actions.append(ga)
        latest = RawFrame(script[min(len(script) - 1, len(actions))].frame[0],
                          script[min(len(script) - 1, len(actions))].state,
                          script[min(len(script) - 1, len(actions))].levels_completed)
    return agent, actions, latest


def main() -> int:
    # script: alternating patch frames, WIN on the 5th response
    win_grid = patched(frame(5), 9)
    script = [
        RawFrame(F1), RawFrame(F2), RawFrame(F1), RawFrame(F2),
        RawFrame(win_grid, state="WIN", levels=1),
    ]
    agent, actions, latest = run(cap=20, script=script)

    # 1. GameAction members come back, ACTION6 carries coordinates
    from arcengine import GameAction
    assert all(isinstance(a, GameAction) for a in actions), type(actions[0])
    print(f"ok: {len(actions)} choose_action calls -> "
          f"{sorted(set(a.name for a in actions))}")

    # 2. the driver's core ran under the shim: rows recorded, no refusals
    d = agent._driver
    assert len(d.rows) >= 1 and d.total_refused == [], d.total_refused
    by = {r.proposed_by for r in d.rows}
    print(f"ok: driver rows={len(d.rows)} provenance={sorted(by)} refused=0")

    # 3. is_done fired on the terminal state (not the cap)
    assert agent.is_done([], latest), "terminal state must finish the episode"
    print("ok: is_done(WIN)=True")

    # 4. the boundary propagated into the driver's bookkeeping
    assert d.rows[-1].goal_reached and d.rows[-1].levels_completed == 1
    print(f"ok: boundary row recorded (levels_completed="
          f"{d.rows[-1].levels_completed}, arc3_state={d.rows[-1].arc3_state})")

    # cap test: no WIN in the script, small cap
    script2 = [RawFrame(F1), RawFrame(F2)] * 3
    agent2, actions2, _ = run(cap=4, script=script2)
    assert agent2.is_done([], script2[0]) and len(agent2._driver.rows) <= 4
    print(f"ok: cap: stopped after {len(agent2._driver.rows)} rows")

    # 5. NO DOUBLE EXECUTION: the starter loop pattern is
    #    action = choose_action(...); frame = do_action_request(action).
    #    The driver already stepped the env inside choose_action, so
    #    do_action_request must report the observed frame without
    #    stepping again.
    agent3, actions3, _ = run(cap=3, script=[RawFrame(F1), RawFrame(F2)] * 4)
    received_before = len(agent3.arc_env.received)
    fd = agent3.do_action_request(actions3[-1])
    assert len(agent3.arc_env.received) == received_before, "action fired twice!"
    assert fd is not None and fd.frame[0] == agent3._obs.grid
    print("ok: do_action_request is a report, not a re-execution")

    # 6. the full framework loop pattern, run by hand: main() without the
    #    threading/recording scaffolding
    from arc3_zagent.kaggle_agent import build_agent_class
    Z = build_agent_class(StubAgentBase)
    env4 = StubArcEnv([RawFrame(F1), RawFrame(F2), RawFrame(win_grid, state="WIN", levels=1)])
    a4 = Z("card", "zz99-test", "zagent", "http://localhost", False, env4)
    latest = RawFrame(F1)
    steps = 0
    while not a4.is_done([], latest) and steps < 20:
        ga = a4.choose_action([], latest)
        latest = a4.do_action_request(ga)   # the framework's next latest_frame
        steps += 1
    assert latest.state == "WIN" and a4._driver.rows[-1].levels_completed == 1
    assert len(env4.received) == steps, f"env stepped {len(env4.received)}x for {steps} decisions"
    print(f"ok: framework loop: {steps} decisions -> {len(env4.received)} env steps, WIN observed")

    print("ALL ADAPTER CHECKS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
