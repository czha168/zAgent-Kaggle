"""R0.1: the reflection-corpus recorder (docs/zagent-rule-reflection-impl-plan.md).

A transparent wrapper around the sidecar client that mirrors every
LEARNING command — register_op / input / observe — to `events.jsonl`,
one row per command, with the driver's step context stamped. The
diagnostics commands (state / stats / ask / snapshot) learn nothing and
are not recorded.

The row shape is the corpus contract (read by adoption/replay_mind.py):

    {"seq": n, "attempt": a, "step": s|null,
     "cmd": {"op": "register_op", "name": op}
           | {"op": "input", "narsese": line, "per_event": n|null, "tail": n|null}
           | {"op": "observe", "act": op, "performed": bool},
     "acts": [op, ...],      # ops the mind proposed in this reply
     "refused": n}

Act ids are transport-local, so observe rows carry the resolved op name
(the proxy keeps an id->op map from each reply's acts). Recording is
append-only and off by default — a driver without `record_path` is
byte-identical to the unrecorded posture (G-R1's live half).
"""

from __future__ import annotations

import json
from pathlib import Path

# learning commands mirror the events the replay feeds; `configure`
# rides along because POSTURE is replay-relevant — a corpus recorded
# under the eviction floor replays with the floor or its inventory
# diverges (M3's A/B is exactly this divergence, measured)
LEARNING_OPS = frozenset({"register_op", "input", "observe", "configure"})


class RecordingMind:
    def __init__(self, inner, path: Path, attempt: int = 0):
        self.inner = inner
        self.path = Path(path)
        self.context: dict = {"attempt": attempt, "step": None}
        self._seq = 0
        self._act_ops: dict[str, str] = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # -- learning commands (recorded) -------------------------------------------

    def call(self, cmd: dict) -> dict:
        reply = self.inner.call(cmd)
        op = cmd.get("op")
        if op == "register_op":
            self._record({"op": "register_op", "name": cmd.get("name")}, reply)
        elif op == "input":
            self._record(self._input_cmd(cmd), reply)
        elif op == "configure":
            self._record({"op": "configure", "floor": cmd.get("eviction_protection_floor"),
                          "lane": cmd.get("operation_goal_lane"),
                          "buffer": cmd.get("event_buffer_capacity")}, reply)
        elif op == "observe":
            act_id = str(cmd.get("act_id"))
            self._record({"op": "observe",
                          "act": self._act_ops.get(act_id, "unknown"),
                          "performed": bool(cmd.get("performed"))}, reply)
        return reply

    def input(self, narsese: str, *, per_event: int | None = None,
              tail: int | None = None) -> dict:
        reply = self.inner.input(narsese, per_event=per_event, tail=tail)
        self._record({"op": "input", "narsese": narsese,
                      "per_event": per_event, "tail": tail}, reply)
        return reply

    def configure(self, **kw) -> dict:
        """Posture write, recorded into the corpus (replay fidelity)."""
        reply = self.inner.configure(**kw)
        self._record({"op": "configure",
                      "floor": kw.get("eviction_protection_floor"),
                      "lane": kw.get("operation_goal_lane"),
                      "buffer": kw.get("event_buffer_capacity")}, reply)
        return reply

    # -- diagnostics commands (passthrough, never recorded) ---------------------

    def cycles(self, n: int) -> dict:
        return self.inner.cycles(n)

    def ask(self, question: str, cycles: int = 50) -> dict:
        return self.inner.ask(question, cycles)

    def state(self) -> dict:
        return self.inner.state()

    def stats(self) -> dict:
        return self.inner.stats()

    def snapshot(self) -> dict:
        return self.inner.snapshot()

    def restore(self, data: bytes) -> dict:
        return self.inner.restore(data)

    def replay(self, bundle: str) -> dict:
        return self.inner.replay(bundle)

    def reset(self) -> dict:
        return self.inner.reset()

    def shutdown(self) -> dict:
        return self.inner.shutdown()

    # -- internals ---------------------------------------------------------------

    @staticmethod
    def _input_cmd(cmd: dict) -> dict:
        return {"op": "input", "narsese": cmd.get("narsese", ""),
                "per_event": cmd.get("per_event"), "tail": cmd.get("tail")}

    def _record(self, cmd: dict, reply: dict) -> None:
        self._seq += 1
        acts = [a.get("op") for a in (reply.get("acts") or [])
                if a.get("op") is not None]
        for a in (reply.get("acts") or []):
            if a.get("id") is not None and a.get("op") is not None:
                self._act_ops[str(a["id"])] = a["op"]
        row = {"seq": self._seq,
               "attempt": self.context.get("attempt"),
               "step": self.context.get("step"),
               "cmd": cmd,
               "acts": acts,
               "refused": len(reply.get("refused") or [])}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
