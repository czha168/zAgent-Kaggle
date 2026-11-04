"""The engine-package import seam: zEngram is the rebrand (2026-11) of
nars-memento, and the arc3 client stack lives in `zengram.envs.arc3`
under the new name. One seam for every module in this package — a
checkout that predates the rebrand falls back to `nars_memento.envs`
with all names intact (verified: Arc3Client, ApiTransport,
ScriptedTransport, Arc3Env, resolve_game_id, Arc3AuthError carry over
verbatim).

`ZENGRAM_SRC` (a src/ checkout dir) is honored before the import
attempt, so a sibling checkout works without installing the wheel.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

_src = (os.environ.get('ZENGRAM_SRC')
        or str(Path(__file__).resolve().parents[2].parent
               / 'zEngram' / 'src'))
if _src and _src not in sys.path:
    sys.path.insert(0, _src)

for _mod in ('zengram.envs.arc3', 'nars_memento.envs.arc3'):
    try:
        envs_arc3 = importlib.import_module(_mod)
        _source = _mod.split('.')[0]
        break
    except ImportError:
        continue
else:
    raise ImportError(
        'no arc3 env module: install the zEngram package (the '
        'nars-memento rebrand) or point ZENGRAM_SRC at a src/ checkout')

# The rebrand's _KeepAliveHTTPSHandler rebuilds the wire request from
# req.unredirected_hdrs ALONE — but user headers passed via the
# `headers=` kwarg ride req.headers, so X-API-Key silently vanishes
# (spy-proven 2026-11-01: the wire carried Host/User-agent/Connection
# only and the server answered 401, while the same key over curl —
# any header case — returns 200). Stock urllib merges both dicts; the
# engine-side fix is the owner's crossing. The client's own
# injectable-opener seam is the designed escape here: a stock opener
# (cookie jar kept for AWSALB session affinity, keep-alive pooling
# forfeited) sends every header.
# The agent name the ARC AGI endpoint receives for every card this
# stack opens: zAgent (the harness), never the engine package's name.
# The API has no dedicated agent field — identity rides the
# scorecard's `tags` (labels the leaderboard filters on) and `opaque`
# (free-form metadata); the client override stamps both on every
# open, and the Arc3Env subclass defaults its tags so a card opened
# without explicit tags still names the agent.
AGENT_NAME = 'zAgent'


if _source == 'zengram':
    from urllib.request import build_opener
    from urllib.request import HTTPCookieProcessor
    from http.cookiejar import CookieJar

    class Arc3Client(envs_arc3.Arc3Client):    # noqa: N801 (seam alias)
        def __init__(self, *args, **kwargs):
            jar = kwargs.get('cookie_jar') or CookieJar()
            kwargs['cookie_jar'] = jar
            kwargs.setdefault(
                'opener',
                build_opener(HTTPCookieProcessor(jar)))
            super().__init__(*args, **kwargs)

        def open_scorecard(self, *, source_url=None, tags=None,
                           opaque=None):
            tags = [AGENT_NAME] + [t for t in (tags or [])
                                   if t != AGENT_NAME]
            opaque = dict(opaque or {})
            opaque.setdefault('agent', AGENT_NAME)
            opaque.setdefault('harness', 'zAgent arc3_zagent')
            return super().open_scorecard(source_url=source_url,
                                          tags=tags, opaque=opaque)

    import inspect

    # The current zEngram Arc3Env is a dataclass without scorecard_tags —
    # card naming happens in the client's open_scorecard override above.
    # Older upstreams took the kwarg; default it only when supported.
    _accepts_tags = 'scorecard_tags' in inspect.signature(
        envs_arc3.Arc3Env.__init__).parameters

    class Arc3Env(envs_arc3.Arc3Env):          # noqa: N801 (seam alias)
        def __init__(self, *args, **kwargs):
            if _accepts_tags:
                tags = kwargs.get('scorecard_tags')
                kwargs['scorecard_tags'] = [AGENT_NAME] + [
                    t for t in (tags or []) if t != AGENT_NAME]
            super().__init__(*args, **kwargs)
else:
    Arc3Client = envs_arc3.Arc3Client
    Arc3Env = envs_arc3.Arc3Env
