"""Allowlisting egress proxy: the runnable audit behind the submission's
no-LLM claim (docs/arc-prize-2026-paper-track-plan.md; the paper's "no
prompt touches the reasoning").

Usage — audit any run's complete egress:

    python -m arc3_zagent.egress_audit &          # logs to /tmp/egress_audit.log
    HTTPS_PROXY=http://127.0.0.1:8899 \\
    HTTP_PROXY=http://127.0.0.1:8899 <run the agent>

Every HTTPS connection the audited process makes is forced through here
(`requests`/urllib honor the proxy env; the mind sidecar makes no
connections at all — see runs/arc3-offline-proof/). Targets not on the
allowlist are refused and logged, never silently passed. The audit's
three-layer form:

1. static: no endpoint strings in the shipped binaries, no LLM-client
   imports in the stack (grep; documented in the package README),
2. negative: a deliberate call to any non-allowlisted host (e.g. an LLM
   endpoint) is BLOCKED and logged,
3. positive: the agent's full live run shows ALLOW lines for
   three.arcprize.org only — measured 5 pooled tunnels for a 309-action
   run, 0 blocked, 0 errors.

For container runs, pass `-e HTTPS_PROXY=http://host.docker.internal:8899`
(`--add-host=host.docker.internal:host-gateway`).
"""
from __future__ import annotations

import socket
import threading
import time

ALLOW = {"three.arcprize.org"}
LOG_PATH = "/tmp/egress_audit.log"
LISTEN = ("0.0.0.0", 8899)


def log(line: str) -> None:
    with open(LOG_PATH, "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {line}\n")


def pump(src: socket.socket, dst: socket.socket) -> None:
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        pass
    finally:
        try:
            dst.shutdown(socket.SHUT_WR)
        except OSError:
            pass


def handle(client: socket.socket) -> None:
    try:
        client.settimeout(30)
        head = b""
        while b"\r\n\r\n" not in head:
            chunk = client.recv(65536)
            if not chunk:
                return
            head += chunk
        req = head.split(b"\r\n")[0].decode("latin1")
        parts = req.split(" ")
        method, target = parts[0], parts[1] if len(parts) > 1 else "?"
        if method == "CONNECT":
            host = target.rsplit(":", 1)[0]
            verdict = "ALLOW" if host in ALLOW else "BLOCK"
            log(f"{verdict} CONNECT {target}")
            if verdict == "BLOCK":
                client.sendall(b"HTTP/1.1 403 forbidden-host\r\n\r\n")
                return
            dest = socket.create_connection(
                (host, int(target.rsplit(":", 1)[1])), timeout=15)
            client.sendall(b"HTTP/1.1 200 tunnel established\r\n\r\n")
        else:
            # absolute-form http:// (non-TLS): extract the host, audit,
            # proxy blind (the agent's traffic is HTTPS; this branch exists
            # so nothing absolute-form slips past unlogged)
            host = target.split("//")[-1].split("/")[0].rsplit(":", 1)[0]
            verdict = "ALLOW" if host in ALLOW else "BLOCK"
            log(f"{verdict} {method} {target}")
            if verdict == "BLOCK":
                client.sendall(b"HTTP/1.1 403 forbidden-host\r\n\r\n")
                return
            dest = socket.create_connection((host, 80), timeout=15)
            dest.sendall(head)
        t1 = threading.Thread(target=pump, args=(client, dest), daemon=True)
        t2 = threading.Thread(target=pump, args=(dest, client), daemon=True)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        dest.close()
    except Exception as e:
        log(f"PROXY-ERROR {e}")
    finally:
        client.close()


def serve(listen: tuple[str, int] = LISTEN) -> None:
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(listen)
    srv.listen(64)
    log(f"proxy listening on {listen[0]}:{listen[1]}, allow={sorted(ALLOW)}")
    while True:
        c, _ = srv.accept()
        threading.Thread(target=handle, args=(c,), daemon=True).start()


if __name__ == "__main__":
    serve()
