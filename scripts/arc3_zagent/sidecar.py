"""JSON-lines client for the `zagent-sidecar` mind process.

One zAgent mind per child process; every command is one blocking round trip
exactly like `MindHandle::call` on the Rust side. The FIFO stdin order is
the total order, so a deterministic command script walks a deterministic
mind timeline (the G-Z1 property, across the process boundary).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

READY_TIMEOUT_S = 30.0
CALL_TIMEOUT_S = 120.0


class SidecarError(RuntimeError):
    pass


def default_sidecar_path() -> Path:
    """`$ZAGENT_SIDECAR`, else the release build, else the debug build."""
    env = os.environ.get("ZAGENT_SIDECAR")
    if env:
        return Path(env)
    repo = Path(__file__).resolve().parents[2]
    for profile in ("release", "debug"):
        candidate = repo / "target" / profile / "zagent-sidecar"
        if candidate.exists():
            return candidate
    raise SidecarError(
        "no zagent-sidecar binary: build it with "
        "`cargo build -p zmind_rpi --bin zagent-sidecar --release` "
        "(or point ZAGENT_SIDECAR at one)"
    )


class MindSidecar:
    """A running `zagent-sidecar` child; context manager = clean shutdown."""

    def __init__(
        self,
        *,
        binary: Path | str | None = None,
        memory: int = 400,
        capacity: int = 100,
        seed: int = 42,
        profile: str = "full",
    ):
        path = Path(binary) if binary else default_sidecar_path()
        if not path.exists() and not shutil.which(str(path)):
            raise SidecarError(f"sidecar binary not found: {path}")
        self.proc = subprocess.Popen(
            [
                str(path),
                "--memory", str(memory),
                "--capacity", str(capacity),
                "--seed", str(seed),
                "--profile", profile,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._next_id = 1
        ready = self._readline(READY_TIMEOUT_S)
        if ready.get("event") != "ready":
            raise SidecarError(f"sidecar did not announce ready: {ready}")
        self.ready_info = ready

    # -- protocol ------------------------------------------------------------------

    def _readline(self, timeout_s: float) -> dict:
        assert self.proc.stdout is not None
        deadline = time.monotonic() + timeout_s
        # readline has no timeout on pipes; the child answers every request,
        # so a hang means a dead/ wedged child — poll exit status as the beat.
        while True:
            line = self.proc.stdout.readline()
            if line:
                try:
                    return json.loads(line)
                except json.JSONDecodeError as e:
                    raise SidecarError(f"unparseable sidecar line: {line!r}") from e
            if self.proc.poll() is not None:
                raise SidecarError(
                    f"sidecar exited (rc={self.proc.returncode}) mid-conversation"
                )
            if time.monotonic() > deadline:
                raise SidecarError(f"sidecar reply timeout after {timeout_s}s")
            time.sleep(0.01)

    def call(self, cmd: dict) -> dict:
        """One round trip; returns the reply payload (without the envelope)."""
        if self.proc.poll() is not None:
            raise SidecarError("sidecar has exited")
        rid = self._next_id
        self._next_id += 1
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps({"id": rid, "cmd": cmd}) + "\n")
        self.proc.stdin.flush()
        msg = self._readline(CALL_TIMEOUT_S)
        if msg.get("error"):
            raise SidecarError(f"sidecar rejected request: {msg['error']}")
        if msg.get("id") != rid:
            raise SidecarError(f"reply id mismatch: want {rid}, got {msg.get('id')}")
        return msg["reply"]

    # -- the mind commands, shaped for the driver ---------------------------------

    def register_op(self, name: str) -> dict:
        return self.call({"op": "register_op", "name": name})

    def input(self, narsese: str, *, per_event: int | None = None, tail: int | None = None) -> dict:
        cmd: dict = {"op": "input", "narsese": narsese}
        if per_event is not None:
            cmd["per_event"] = per_event
            if tail is not None:
                cmd["tail"] = tail
        return self.call(cmd)

    def cycles(self, n: int) -> dict:
        return self.call({"op": "cycles", "n": n})

    def configure(self, *, operation_goal_lane: bool | None = None,
                  event_buffer_capacity: int | None = None,
                  eviction_protection_floor: float | None = None,
                  concept_headroom: int | None = None) -> dict:
        """Host-tunable posture writes (the sidecar's Configure): the
        eviction floor is the M3-measured lever (2.9x mined rules, 55
        over the 0.4 bar — the consolidation posture for any run whose
        mined rule set is the asset). concept_headroom is the restore
        opener (the PTCG restored-inertness counter); the engine
        reports the eviction in the reply line, not in stats.lane."""
        cmd: dict = {"op": "configure"}
        if operation_goal_lane is not None:
            cmd["operation_goal_lane"] = operation_goal_lane
        if event_buffer_capacity is not None:
            cmd["event_buffer_capacity"] = event_buffer_capacity
        if eviction_protection_floor is not None:
            cmd["eviction_protection_floor"] = eviction_protection_floor
        if concept_headroom is not None:
            cmd["concept_headroom"] = concept_headroom
        return self.call(cmd)

    def ask(self, question: str, cycles: int = 50) -> dict:
        return self.call({"op": "ask", "question": question, "cycles": cycles})

    def state(self) -> dict:
        return self.call({"op": "state"})

    def stats(self) -> dict:
        """The read-only lane-stats digest (A2 probe): predictions
        {open, dropped_resolutions, tracks[]} and mined_rules {count,
        top[]} — no cycles charged, no memory moved (G-CL5 pins the
        snapshot bytes across it)."""
        return self.call({"op": "stats"})

    def snapshot(self) -> dict:
        return self.call({"op": "snapshot"})

    def restore(self, data: bytes) -> dict:
        """Restore snapshot bytes into the fresh engine (the warm start).
        The op registry is NOT in the snapshot — the host re-registers ops
        after this (the driver's `_register_ops` at episode start does)."""
        return self.call({"op": "restore", "bytes": list(data)})

    def replay(self, bundle: str) -> dict:
        """Replay an epoch bundle as consolidation: a bounded sample of
        its judgements re-enters as low-budget `:|:` events, so temporal
        induction mines ACROSS episodes. Needs the FULL profile — Parity
        refuses with `replay consolidation is off`."""
        return self.call({"op": "replay", "bundle": bundle})

    def reset(self) -> dict:
        """Fresh engine state; the op registry survives (it is bridge-side)."""
        return self.call({"op": "reset"})

    def shutdown(self) -> dict:
        try:
            reply = self.call({"op": "shutdown"})
        except (SidecarError, BrokenPipeError):
            reply = {"lines": ["already gone"]}
        self.proc.wait(timeout=10)
        return reply

    # -- lifecycle -------------------------------------------------------------------

    def __enter__(self) -> "MindSidecar":
        return self

    def __exit__(self, *exc) -> None:
        try:
            self.shutdown()
        except Exception:
            self.proc.kill()
