"""CLI: `python -m arc3_zagent --game ls20 [--actions 40] [--workspace runs/…]`

Live runs talk to three.arcprize.org (`ARC3_API_KEY` required; the scorecard
card persists into `<workspace>/arc3_session.json` per the engine's
lesson). `--scripted <fixture.json>` replaces the network with a canned
FrameResponse list (the offline gate); `--zengram-src` overrides where
the engine package (zEngram, the nars-memento rebrand) is found
(default: the installed wheel, else the sibling checkout).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .driver import Arc3Driver
from .sidecar import MindSidecar, default_sidecar_path


def build_env(args):
    from ._compat import Arc3Client as SeamArc3Client
    from ._compat import Arc3Env as SeamArc3Env
    from ._compat import envs_arc3

    ApiTransport, ScriptedTransport = (
        envs_arc3.ApiTransport, envs_arc3.ScriptedTransport)
    Arc3Client = SeamArc3Client   # the rebrand transport drops X-API-Key
    Arc3Env = SeamArc3Env         # scorecards name the agent: zAgent

    if args.scripted:
        responses = json.loads(Path(args.scripted).read_text(encoding="utf-8"))
        return Arc3Env(args.game, ScriptedTransport(responses))
    client = Arc3Client()  # ARC3_API_KEY / ARC3_BASE_URL from the environment
    workspace = Path(args.workspace) if args.workspace else None
    session_path = workspace / "arc3_session.json" if workspace else None
    return Arc3Env(
        args.game,
        ApiTransport(client, session_path=session_path),
        game_id_resolver=lambda spec: _resolve(client, spec),
    )


def _resolve(client, spec: str) -> str:
    from ._compat import envs_arc3

    resolve_game_id = envs_arc3.resolve_game_id

    return resolve_game_id(client.list_games(), spec)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="arc3_zagent", description="play ARC-AGI-3 through the zAgent mind"
    )
    parser.add_argument("--game", required=True, help="slug or versioned game id (e.g. ls20)")
    parser.add_argument("--actions", type=int, default=40, help="action cap (default 40)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--memory", type=int, default=400)
    parser.add_argument("--capacity", type=int, default=100)
    parser.add_argument("--profile", choices=["full", "parity"], default="full")
    parser.add_argument("--workspace", default=None, help="run artifacts dir")
    parser.add_argument(
        "--warm-start", default=None, help="restore this mind snapshot (from a previous episode) before playing"
    )
    parser.add_argument(
        "--record-events", action="store_true",
        help="mirror every learning command to <workspace>/events.jsonl "
             "(the reflection corpus, R0.1 of the reflection plan)",
    )
    parser.add_argument("--scripted", default=None, help="offline: JSON list of FrameResponses")
    parser.add_argument("--sidecar", default=None, help="path to zagent-sidecar binary")
    parser.add_argument(
        "--zengram-src", default=None, help="zEngram src dir (default: installed wheel, else sibling checkout)"
    )
    args = parser.parse_args(argv)

    if args.zengram_src:
        import os

        os.environ["ZENGRAM_SRC"] = args.zengram_src

    from . import percepts  # after the override, if any

    if str(percepts.NARS_MEMENTO_SRC) not in sys.path:
        sys.path.insert(0, str(percepts.NARS_MEMENTO_SRC))

    kwargs = {}
    if args.sidecar:
        kwargs["binary"] = args.sidecar
    with MindSidecar(
        memory=args.memory, capacity=args.capacity, seed=args.seed, profile=args.profile, **kwargs
    ) as mind:
        env = build_env(args)
        record_path = (
            Path(args.workspace) / "events.jsonl"
            if args.record_events and args.workspace else None
        )
        driver = Arc3Driver(env, mind, seed=args.seed, record_path=record_path)
        if args.warm_start:
            driver.warm_start(Path(args.warm_start))
        result = driver.run(max_actions=args.actions, workspace=Path(args.workspace) if args.workspace else None)
        print(json.dumps({k: v for k, v in result.summary.items() if k != "change_probe"}, indent=2))
        probe = result.summary["change_probe"]
        if any(probe.values()):
            print("change beliefs:")
            for op, answers in probe.items():
                for a in answers:
                    print(f"  {op}: {a}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
