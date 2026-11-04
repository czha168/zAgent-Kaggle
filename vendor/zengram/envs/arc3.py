"""ARC-AGI-3 adapter (Phase 5, proposal §3/§4.1/§6): `arc3.py`.

ARC-AGI-3 (three.arcprize.org) is the paper's headline benchmark: 64×64
symbolic grids, deterministic games, goals acquired on the fly. The REST API
(baseline1-compatible I/O, per the OpenAPI spec at docs.arcprize.org) is:

    POST /api/cmd/RESET            {game_id, card_id, guid?}   → FrameResponse
    POST /api/cmd/ACTION1..ACTION7 {game_id, guid[, x, y]}     → FrameResponse
    GET  /api/games | /api/games/{id}          game discovery
    POST /api/scorecard/open | /scorecard/close, GET /api/scorecard/{id}

Auth is an `X-API-Key` header; AWSALB cookies must be replayed for session
affinity (the client keeps a cookie jar). A `FrameResponse.frame` is one or
more consecutive 64×64 grids of 4-bit colour indices (0–15) — the extra
frames are internal animation settling, so the adapter's **frame-skip** rule
is: always take the LAST frame of the response (the settled state).

Frame → grid: each colour index maps to a single-character symbol (default:
hex digits `0`–`f`, injective over the 16 values), yielding the symbolic
grid the replay gate compares cell-exactly. No downsampling is needed — the
API's frame is already the 64×64 grid the paper assumes.

Episodes and goals: levels advance automatically ( FrameResponse
`levels_completed` increments); `WIN` = every level cleared, `GAME_OVER` =
level attempt exhausted. `goal_reached` marks an EPISODE BOUNDARY for the
learning loop — True on level completion, `WIN`, or `GAME_OVER` — mirroring
how gridworld episodes end at the goal. True wins vs game-overs stay
distinguishable in `Obs.info["arc3_state"]` and, authoritatively, on the
server scorecard (`metrics/rhae.py` scores from that, never from our flag).

ACTION6 is the one coordinate action (x, y ∈ 0..63): `step("ACTION6@x,y")`
executes it. It is not advertised verbatim — 4096 candidate actions would
blow up exploration and search. Games mixing ACTION6 with simple actions
advertise only the simple ones unless `advertise_action6` passes a
coordinate sample; click-only games (available_actions == [6], e.g. FT09)
auto-advertise a small deterministic sample so their interface is never
empty.

Everything runs on the stdlib (urllib + cookie jar): the core install stays
dependency-free. Offline tests/recording use `ScriptedTransport`; real runs
use `ApiTransport` over `Arc3Client`.
"""

from __future__ import annotations

import http.client
import json
import os
import select
from dataclasses import dataclass, field
from http.cookiejar import Cookie, CookieJar
from pathlib import Path
from typing import Callable, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import HTTPSHandler, Request, build_opener
from urllib.request import HTTPCookieProcessor

from .base import Obs

ARC3_GRID = 64  # frames are 64×64
ARC3_COLORS = 16  # 4-bit colour indices, 0–15
ARC3_API_KEY_ENV = "ARC3_API_KEY"
ARC3_BASE_URL_ENV = "ARC3_BASE_URL"
DEFAULT_BASE_URL = "https://three.arcprize.org"
RESET_ACTION_ID = 0  # action_input.id for RESET; never an episode action
ACTION6_ID = 6  # the one coordinate action (x, y ∈ 0..63)
ACTION6 = "ACTION6"
# ACTION1..5, ACTION7 are simple (no coordinates); 6 is the coordinate action.
SIMPLE_ACTION_IDS = (1, 2, 3, 4, 5, 7)


class Arc3Error(RuntimeError):
    """Any ARC-AGI-3 transport/protocol failure."""


class Arc3AuthError(Arc3Error):
    """Missing or invalid API key (HTTP 401)."""


class Arc3ApiError(Arc3Error):
    """The API answered with an error status."""


# --- frame → grid -------------------------------------------------------------------


def default_palette() -> list[str]:
    """Hex digits: 16 colour indices → 16 distinct single-char symbols."""
    return [f"{i:x}" for i in range(ARC3_COLORS)]


def frame_to_grid(
    frame: list[list[int]], palette: list[str] | None = None
) -> list[list[str]]:
    """Map one 64×64 frame of colour indices to the symbolic grid.

    Strict on purpose: a shape/value drift means the replay gate would be
    comparing against a differently-encoded world — fail loudly instead.
    """
    symbols = palette if palette is not None else default_palette()
    if len(frame) != ARC3_GRID or any(len(row) != ARC3_GRID for row in frame):
        raise Arc3Error(
            f"expected a {ARC3_GRID}×{ARC3_GRID} frame, got "
            f"{len(frame)}×{len(frame[0]) if frame else 0}"
        )
    grid: list[list[str]] = []
    for row in frame:
        out_row: list[str] = []
        for cell in row:
            if not isinstance(cell, int) or not 0 <= cell < ARC3_COLORS:
                raise Arc3Error(f"colour index out of range 0–{ARC3_COLORS - 1}: {cell!r}")
            out_row.append(symbols[cell])
        grid.append(out_row)
    return grid


def action_name(action_id: int) -> str:
    if action_id == ACTION6_ID:
        return ACTION6
    if action_id in SIMPLE_ACTION_IDS:
        return f"ACTION{action_id}"
    raise Arc3Error(f"unknown action id {action_id!r}")


def coordinate_sample(step: int = 16) -> list[tuple[int, int]]:
    """Deterministic coordinate sample for ACTION6-only (click) games.

    Step 16 gives a 4×4 grid = 16 candidate clicks — bounded so random
    exploration and search stay usable on games like FT09 that advertise
    no simple actions at all."""
    return [(x, y) for y in range(0, ARC3_GRID, step) for x in range(0, ARC3_GRID, step)]


def parse_action(action: str) -> tuple[str, int | None, int | None]:
    """`"ACTION6@x,y"` → ("ACTION6", x, y); simple actions → (name, None, None)."""
    name, sep, coords = action.partition("@")
    if not sep:
        return action, None, None
    parts = coords.split(",")
    if name != ACTION6 or len(parts) != 2:
        raise ValueError(f"malformed coordinate action {action!r} (want 'ACTION6@x,y')")
    x, y = int(parts[0]), int(parts[1])
    if not (0 <= x < ARC3_GRID and 0 <= y < ARC3_GRID):
        raise ValueError(f"ACTION6 coordinates out of range 0–{ARC3_GRID - 1}: {action!r}")
    return name, x, y


def validate_frame_response(resp: dict) -> None:
    """Structural check of a FrameResponse (fail fast on API drift)."""
    required = ("game_id", "guid", "frame", "state", "levels_completed", "win_levels")
    missing = [key for key in required if key not in resp]
    if missing:
        raise Arc3Error(f"FrameResponse missing fields {missing}: keys={sorted(resp)}")
    frames = resp["frame"]
    if not isinstance(frames, list) or not frames:
        raise Arc3Error("FrameResponse.frame must be a non-empty list of frames")
    for frame in frames:
        if not isinstance(frame, list) or len(frame) != ARC3_GRID or any(
            not isinstance(row, list) or len(row) != ARC3_GRID for row in frame
        ):
            raise Arc3Error(
                f"FrameResponse frames must be {ARC3_GRID}×{ARC3_GRID} grids"
            )


def available_action_names(resp: dict) -> list[str]:
    """Episode actions for this point in the game: ACTION ids minus RESET.

    RESET (id 0) is env-managed (it starts/restarts episodes), never part of
    the agent's in-episode action interface."""
    names: list[str] = []
    for action_id in resp.get("available_actions") or []:
        if action_id == RESET_ACTION_ID:
            continue
        names.append(action_name(action_id))
    return names


# --- transports -----------------------------------------------------------------------


class Transport(Protocol):
    """The network seam: everything an offline test must stand in for."""

    def reset(self, game_id: str, guid: str | None) -> dict: ...
    def action(
        self, game_id: str, guid: str | None, name: str, x: int | None, y: int | None
    ) -> dict: ...


class _KeepAliveHTTPSHandler(HTTPSHandler):
    """HTTPS handler that keeps one pooled connection per client.

    The stock handler sends `Connection: close` and drops the socket after
    every request, so each call pays a full TCP connect + TLS handshake —
    measured at ~40% of the grind loop's wall time to CloudFront. This
    handler reuses the connection; before sending on it, a zero-timeout
    readability probe discards sockets the peer closed (or that still hold
    undrained bytes from a previous response) so staleness never fails a
    request. Single-threaded, like its callers. Send/response failures
    surface as URLError — the stock handler's error contract, so
    `Arc3Client._request`'s mapping is unchanged.
    """

    def __init__(self, context=None):
        super().__init__(context=context)
        self._conn: http.client.HTTPConnection | None = None

    def close_pooled(self) -> None:
        """Drop the pooled connection (client shutdown)."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def https_open(self, req):
        return self._do_open(req)

    @staticmethod
    def _unusable(sock) -> bool:
        # Readable before we sent anything means EOF (peer closed) or
        # unsolicited bytes: either disqualifies the connection.
        try:
            return bool(select.select([sock], [], [], 0)[0])
        except (OSError, ValueError):
            return True

    def _do_open(self, req):
        conn = self._conn
        if conn is not None and (conn.sock is None or self._unusable(conn.sock)):
            conn.close()
            conn = None
        if conn is None:
            conn = self._conn = http.client.HTTPSConnection(
                req.host, timeout=req.timeout, context=self._context
            )
        elif isinstance(req.timeout, (int, float)):
            conn.sock.settimeout(req.timeout)
        headers = dict(req.unredirected_hdrs)
        headers["Connection"] = "keep-alive"
        try:
            conn.request(req.get_method(), req.selector, req.data, headers)
            resp = conn.getresponse()
        except (OSError, http.client.HTTPException) as exc:
            self._conn = None
            conn.close()
            raise URLError(exc) from None
        resp.url = req.get_full_url()
        resp.msg = resp.reason
        return resp


class Arc3Client:
    """Baseline1-compatible REST client. stdlib-only; keeps AWSALB cookies;
    reuses one HTTPS connection across requests.

    `opener` is injectable for tests; real callers get an opener with a
    cookie jar (session affinity is REQUIRED by the API for game state).
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        timeout_s: float = 60.0,
        opener=None,
        cookie_jar: CookieJar | None = None,
    ):
        self.api_key = api_key or os.environ.get(ARC3_API_KEY_ENV) or ""
        if not self.api_key:
            raise Arc3AuthError(
                f"no ARC-AGI-3 API key: pass api_key or set {ARC3_API_KEY_ENV}"
            )
        self.base_url = (base_url or os.environ.get(ARC3_BASE_URL_ENV) or DEFAULT_BASE_URL).rstrip("/")
        self.timeout_s = timeout_s
        # AWSALB cookies are load-bearing beyond game state: scorecard lookups
        # are affinity-bound too (live-run lesson — a card 404s from a fresh
        # client after its creating process exits). The jar is exposed so the
        # session file can persist and restore it.
        self.cookie_jar = cookie_jar or CookieJar()
        self._keep_alive = _KeepAliveHTTPSHandler() if opener is None else None
        self._opener = opener or build_opener(
            HTTPCookieProcessor(self.cookie_jar), self._keep_alive
        )

    def close(self) -> None:
        """Release pooled keep-alive connections held by this client or by an
        injected opener. Harmless twice."""
        handlers = getattr(self._opener, "handlers", [self._keep_alive])
        for handler in handlers:
            close_pooled = getattr(handler, "close_pooled", None)
            if close_pooled is not None:
                close_pooled()

    def _request(self, path: str, payload: dict | None = None):
        body = None
        headers = {"X-API-Key": self.api_key}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(
            f"{self.base_url}{path}", data=body, headers=headers,
            method="GET" if payload is None else "POST",
        )
        try:
            with self._opener.open(request, timeout=self.timeout_s) as http_resp:
                raw = http_resp.read()
        except HTTPError as exc:
            detail = exc.read()[:500].decode("utf-8", errors="replace")
            if exc.code == 401:
                # The body says WHY: a revoked key and throttle-by-auth
                # both answer 401 — keep the detail so logs can tell them
                # apart (live lesson, 2026-10-27 ls20 grind).
                raise Arc3AuthError(f"{path}: HTTP 401: {detail}") from None
            raise Arc3ApiError(f"{path}: HTTP {exc.code}: {detail}") from None
        except URLError as exc:
            raise Arc3Error(f"{path}: {exc.reason}") from None
        if not raw:
            return None
        return json.loads(raw)

    # -- games ------------------------------------------------------------------

    def list_games(self) -> list[dict]:
        return self._request("/api/games")

    def game_info(self, game_id: str) -> dict:
        return self._request(f"/api/games/{game_id}")

    # -- scorecards ----------------------------------------------------------------

    def open_scorecard(self, *, source_url=None, tags=None, opaque=None) -> str:
        payload = {k: v for k, v in
                   {"source_url": source_url, "tags": tags, "opaque": opaque}.items()
                   if v is not None}
        return self._request("/api/scorecard/open", payload or {})["card_id"]

    def close_scorecard(self, card_id: str) -> dict:
        return self._request("/api/scorecard/close", {"card_id": card_id})

    def get_scorecard(self, card_id: str, game_id: str | None = None) -> dict:
        path = f"/api/scorecard/{card_id}"
        if game_id:
            path += f"/{game_id}"
        return self._request(path)

    # -- commands ---------------------------------------------------------------------

    def reset(self, game_id: str, card_id: str, guid: str | None = None) -> dict:
        payload: dict = {"game_id": game_id, "card_id": card_id}
        if guid:
            payload["guid"] = guid
        return self._request("/api/cmd/RESET", payload)

    def action(
        self,
        game_id: str,
        card_id: str,
        guid: str,
        name: str,
        x: int | None = None,
        y: int | None = None,
    ) -> dict:
        if name == ACTION6:
            if x is None or y is None:
                raise ValueError("ACTION6 needs x, y coordinates")
            payload: dict = {"game_id": game_id, "guid": guid, "x": x, "y": y}
        else:
            if name not in {f"ACTION{i}" for i in SIMPLE_ACTION_IDS}:
                raise ValueError(f"unknown ARC-AGI-3 action {name!r}")
            payload = {"game_id": game_id, "guid": guid}
        return self._request(f"/api/cmd/{name}", payload)


# --- scorecard session persistence ---------------------------------------------------


def session_cookies(client) -> list[dict]:
    """Serialise a client's cookie jar for the session file (AWSALB affinity
    cookies are required to fetch a card after the creating process exits)."""
    jar = getattr(client, "cookie_jar", None)
    if jar is None:
        return []
    return [
        {"name": c.name, "value": c.value, "domain": c.domain, "path": c.path}
        for c in jar
        if not c.is_expired()
    ]


def restore_session_cookies(client, record: dict) -> None:
    """Best-effort: put saved affinity cookies back into the client's jar."""
    jar = getattr(client, "cookie_jar", None)
    if jar is None:
        return
    for cookie in record.get("cookies") or []:
        try:
            domain = cookie["domain"]
            jar.set_cookie(
                Cookie(
                    version=0, name=cookie["name"], value=cookie["value"],
                    port=None, port_specified=False,
                    domain=domain, domain_specified=True,
                    domain_initial_dot=domain.startswith("."),
                    path=cookie.get("path", "/"), path_specified=True,
                    secure=False, expires=None, discard=True,
                    comment=None, comment_url=None, rest={},
                )
            )
        except (KeyError, TypeError):
            continue


class ApiTransport:
    """Transport over `Arc3Client`; opens (or reuses) a scorecard lazily.

    `session_path` persists `{card_id, guid, game_id}` so the
    `arc3-scorecard` CLI can later fetch/close the card without the operator
    copying IDs around. Pass an existing `card_id` to keep scoring inside a
    card you opened elsewhere."""

    def __init__(
        self,
        client: Arc3Client,
        *,
        card_id: str | None = None,
        session_path: Path | str | None = None,
        scorecard_tags: list[str] | None = None,
        source_url: str | None = None,
    ):
        self.client = client
        self.card_id = card_id
        self.session_path = Path(session_path) if session_path else None
        self.scorecard_tags = scorecard_tags
        self.source_url = source_url
        self._adopt_session()

    def _adopt_session(self) -> None:
        """Resume an existing card when none was given: a resumed run keeps
        scoring into the same scorecard, and its affinity cookies come back
        with it (without them the card 404s — affinity is server-side)."""
        if self.card_id is not None or self.session_path is None or not self.session_path.exists():
            return
        try:
            record = json.loads(self.session_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        restore_session_cookies(self.client, record)
        self.card_id = record.get("card_id") or None

    def _ensure_card(self) -> str:
        if self.card_id is None:
            self.card_id = self.client.open_scorecard(
                source_url=self.source_url, tags=self.scorecard_tags
            )
            self._save_session()
        return self.card_id

    def _save_session(self, *, game_id: str | None = None, guid: str | None = None) -> None:
        if self.session_path is None:
            return
        self.session_path.parent.mkdir(parents=True, exist_ok=True)
        record = {}
        if self.session_path.exists():
            try:
                record = json.loads(self.session_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                record = {}
        record.update(
            {
                "card_id": self.card_id,
                "game_id": game_id,
                "guid": guid,
                "cookies": session_cookies(self.client),
            }
        )
        self.session_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    def reset(self, game_id: str, guid: str | None) -> dict:
        resp = self.client.reset(game_id, self._ensure_card(), guid)
        self._save_session(game_id=game_id, guid=resp["guid"])
        return resp

    def action(
        self, game_id: str, guid: str | None, name: str, x: int | None, y: int | None
    ) -> dict:
        if not guid:
            raise Arc3Error("no session guid: reset() must run before action()")
        return self.client.action(game_id, self.card_id or "", guid, name, x=x, y=y)


class ScriptedTransport:
    """Offline transport: pops canned FrameResponses in call order.

    The exactly-what-was-scripted stance keeps the env's protocol testable
    without network; `calls` records what the env issued (the same assertable
    record the replay gate expects from real runs)."""

    def __init__(self, responses: list[dict]):
        self.responses = list(responses)
        self.calls: list[tuple] = []

    def _next(self) -> dict:
        if not self.responses:
            raise Arc3Error("scripted transport exhausted: no canned response left")
        return self.responses.pop(0)

    def reset(self, game_id: str, guid: str | None) -> dict:
        self.calls.append(("reset", game_id, guid))
        return self._next()

    def action(
        self, game_id: str, guid: str | None, name: str, x: int | None, y: int | None
    ) -> dict:
        self.calls.append(("action", game_id, guid, name, x, y))
        return self._next()


# --- the Env ----------------------------------------------------------------------------


@dataclass
class Arc3Env:
    """One ARC-AGI-3 game as an `Env`: structured 64×64 observations.

    - reset() starts a fresh game (guid=None) or, after a `GAME_OVER`,
      retries the current level (same guid); after `WIN` the next reset
      starts a new game. A reset issued while a reset response is already
      current returns the cached frame instead of burning a server call
      (the state machine calls reset for signatures between episodes).
    - step() issues ACTION1..7 (ACTION6 as "ACTION6@x,y"), takes the settled
      last frame of the response, and flags the episode boundary on level
      completion, WIN, or GAME_OVER.
    """

    game_id: str
    transport: Transport
    palette: list[str] | None = None
    advertise_action6: list[tuple[int, int]] | None = None  # None = auto for click-only games
    actions: list[str] = field(default_factory=list)
    # Optional slug → versioned-id resolver, invoked once on the first
    # server-touching reset (keeps construction offline; see resolve_game_id).
    game_id_resolver: Callable[[str], str] | None = None

    def __post_init__(self):
        self._guid: str | None = None
        self._levels = 0
        self._steps = 0
        self._game_finished = False  # WIN seen: next reset starts a fresh game
        self._current_reset: dict | None = None  # last RESET response, None once stepped

    # -- Env protocol ----------------------------------------------------------

    def reset(self) -> Obs:
        if self._current_reset is not None:
            # A RESET response is already current (no step since): e.g. the
            # state machine re-signing the env between episodes. Return it
            # instead of burning a server call.
            return self._obs(self._current_reset, steps=0)
        if self.game_id_resolver is not None:
            self.game_id = self.game_id_resolver(self.game_id)
            self.game_id_resolver = None
        guid = None if self._game_finished else self._guid
        try:
            resp = self.transport.reset(self.game_id, guid)
        except Arc3ApiError:
            if guid is None:
                raise
            # The server expires idle game instances (observed after a slow
            # operator session): the stale guid 404s as "game not found".
            # Start a fresh instance instead of dying.
            self._guid = None
            self._levels = 0
            resp = self.transport.reset(self.game_id, None)
        self._ingest(resp)
        self._current_reset = resp
        return self._obs(resp, steps=0)

    def step(self, action: str) -> Obs:
        if self._current_reset is None and self._steps == 0 and self._guid is None:
            raise Arc3Error("step() before reset(): run reset() first")
        if action not in self.actions:
            raise ValueError(
                f"unknown action {action!r}; expected one of {self.actions}"
            )
        name, x, y = parse_action(action)
        self._current_reset = None  # stepped: the next reset must hit the server
        resp = self.transport.action(self.game_id, self._guid, name, x, y)
        previous_levels = self._levels
        self._ingest(resp)
        self._steps += 1
        leveled = resp["levels_completed"] > previous_levels
        boundary = leveled or resp["state"] in ("WIN", "GAME_OVER")
        return self._obs(resp, steps=self._steps, goal=boundary)

    # -- internals ------------------------------------------------------------

    def _ingest(self, resp: dict) -> None:
        validate_frame_response(resp)
        self._guid = resp["guid"]
        self._levels = resp["levels_completed"]
        names = available_action_names(resp)
        # A bare ACTION6 is meaningless without coordinates: advertise it only
        # as sampled "ACTION6@x,y" strings. None (the default) auto-samples a
        # small coordinate grid when the game has NO simple actions — click
        # games like FT09 advertise [6] alone, and an empty interface would
        # stall every policy; [] suppresses sampling entirely.
        has_action6 = ACTION6_ID in (resp.get("available_actions") or [])
        while ACTION6 in names:
            names.remove(ACTION6)
        if self.advertise_action6:
            if has_action6:
                names += [f"{ACTION6}@{x},{y}" for x, y in self.advertise_action6]
        elif self.advertise_action6 is None and has_action6 and not names:
            names += [f"{ACTION6}@{x},{y}" for x, y in coordinate_sample()]
        self.actions = names
        if resp["state"] == "WIN":
            self._game_finished = True

    def _obs(self, resp: dict, *, steps: int, goal: bool | None = None) -> Obs:
        if goal is None:  # reset frames: only a terminal state is a boundary
            goal = resp["state"] in ("WIN", "GAME_OVER")
        settled = frame_to_grid(resp["frame"][-1], self.palette)
        return Obs(
            grid=settled,
            goal_reached=goal,
            steps=steps,
            info={
                "arc3_state": resp["state"],
                "levels_completed": resp["levels_completed"],
                "win_levels": resp["win_levels"],
                "frames_in_response": len(resp["frame"]),  # frame-skip: last one wins
                "full_reset": resp.get("full_reset"),
                "game_finished": self._game_finished,
            },
        )


# --- scorecard snapshot (CLI helper) ----------------------------------------------------


def save_scorecard_snapshot(
    workspace: Path,
    client,  # Arc3Client or duck-typed stand-in
    *,
    card_id: str | None = None,
    close: bool = False,
    game_id: str | None = None,
    session_file: str = "arc3_session.json",
    snapshot_file: str = "arc3_scorecard.json",
) -> dict:
    """Fetch (optionally close) the scorecard and freeze it in the workspace.

    `card_id` falls back to the session file `ApiTransport` wrote during the
    run. The snapshot (`arc3_scorecard.json`) is what `report` renders the
    RHAE/actions-ratio tables from."""
    workspace = Path(workspace)
    if card_id is None:
        session = workspace / session_file
        if session.exists():
            try:
                record = json.loads(session.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                record = {}
            card_id = record.get("card_id")
            restore_session_cookies(client, record)  # affinity: without these, 404
    if not card_id:
        raise Arc3Error(
            f"no card_id: pass --card-id or run the game first (session file "
            f"{workspace / session_file} is written by ApiTransport)"
        )
    summary = (
        client.close_scorecard(card_id) if close else client.get_scorecard(card_id, game_id)
    )
    from datetime import datetime, timezone

    snapshot = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "closed": close,
        "scorecard": summary,
    }
    (workspace / snapshot_file).write_text(
        json.dumps(snapshot, indent=2), encoding="utf-8"
    )
    return summary


def resolve_game_id(games: list[dict], game_id: str) -> str:
    """Normalize a game spec ('ls20' or 'ls20-9607627b') to the versioned id.

    The server resolves slugs for /api/cmd/RESET but NOT for /api/cmd/ACTION*,
    so a slug-named env resets fine and then fails on its first action; the
    env always plays under the versioned id."""
    for game in games:
        if game.get("game_id") == game_id:
            return game_id
    matches = [
        str(game["game_id"])
        for game in games
        if str(game.get("game_id", "")).startswith(f"{game_id}-")
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"unknown ARC-AGI-3 game {game_id!r}")
    raise ValueError(
        f"ambiguous game {game_id!r}; pass the full id (one of {matches})"
    )


def make_arc3_env(
    game_id: str, session_path: Path | str | None = None
) -> "Arc3Env":
    """Online env wired from the environment (`ARC3_API_KEY`, `ARC3_BASE_URL`).

    Pass `session_path` (the run workspace's `arc3_session.json`) so the
    scorecard card survives the process: without it the card id exists only
    server-side and `arc3-scorecard --workspace` can never freeze the RHAE
    tables (the ls20-actor and ls20-investigate cards were lost exactly this
    way)."""
    client = Arc3Client(base_url=os.environ.get(ARC3_BASE_URL_ENV))

    def _resolve(spec: str) -> str:
        return resolve_game_id(client.list_games(), spec)

    return Arc3Env(
        game_id,
        ApiTransport(client, session_path=session_path),
        game_id_resolver=_resolve,
    )
