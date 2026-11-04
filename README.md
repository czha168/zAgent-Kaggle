# zAgent — Kaggle submission release

The reproducibility artifact for the ARC Prize 2026 paper
*"Reasoning Mind without Gradients that Continually Learns and Adapts"*:
the agent harness around the zMind reasoning engine, as submitted to the
ARC-AGI-3 Kaggle competition.

**No LLM anywhere.** The mind is a local deterministic Rust process; the
Python stack is stdlib-only except the ARC tool wheels; a socket-guarded
run of the whole package completed a full episode with zero network
attempts, and the live run's complete egress (audited through an
allowlisting proxy) was 5 pooled tunnels to `three.arcprize.org` only.

## Layout

| path | what it is |
|---|---|
| `scripts/arc3_zagent/` | the Python agent stack: `kaggle_agent.py` (the `Agent`-contract adapter), `driver.py` (per-step loop, plan lane, route-locked posture, in-place level recovery), `percepts.py` (frame → Narsese percepts), `sidecar.py` (the mind's stdio client), `adapter_test.py` (the 7-check offline contract test), `egress_audit.py` (the allowlisting egress proxy) |
| `vendor/zengram/` | the six zEngram modules the stack imports (envs/base, envs/arc3, nars_bridge/{evidence,rules,truth}) — stdlib-only, LLM-free; zEngram's LLM machinery is *not* part of this release |
| `routes/all_25_optimal_routes.json` | the 25 proven-optimal routes (SDK receipts), consumed by the plan lane |
| `zagent_submission.ipynb` | the submitted Kaggle notebook (cell plan validated end-to-end) |
| `receipts/` | machine-checkable run receipts (offline proof, live one-card summary, the paper) |

## The mind (not part of this release)

The reasoning engine itself — zMind, a Non-Axiomatic Logic rule base over
a truth-managed ledger — ships as the **prebuilt, binary-audited ELF**
`zagent-sidecar-linux-x86_64` in the public Kaggle dataset
[`czha168/zagent-package`](https://www.kaggle.com/datasets/czha168/zagent-package)
(CC0, together with the warm-start snapshot `mind_snapshot.bin`). The
binary audit: zero endpoint strings of any kind, no LLM client code
linked (the `rpi-ai` provider feature is compiled out), functions with no
environment variables set. The engine's architecture is described in the
attached paper; its source is not part of this release.

## Reproduce the offline contract

The 7-check contract test (GameAction round-trips, driver under the
shim, `is_done` on cap and terminal, boundary propagation, no
double-execution, the framework loop) runs fully offline. Requirements:
Python 3.11+, `arcengine==0.9.3` and `pydantic` from pip, and the
prebuilt sidecar from the dataset:

```sh
export ZAGENT_SIDECAR=/path/to/zagent-sidecar-linux-x86_64
export ZENGRAM_SRC=$PWD/vendor
PYTHONPATH=$PWD/scripts python3 scripts/arc3_zagent/adapter_test.py
# → ALL ADAPTER CHECKS PASS
```

Determinism: the mind is seeded and single-threaded; the same action
sequence reproduces the same transcript (verified byte-identical across
independent processes at engine 1.5.4 lockstep).

## The submitted run

The Kaggle notebook plays the ARC-AGI-3 games offline through the local
`arc_agi` environments with the warm-start snapshot and the routebook
plan lane. Receipts:

- **Live one-card sweep**: scorecard `8a7b0299-f736-481b-be9e-f93da1259c18`
  — 25/25 games WIN, all levels, 5,502 actions (exactly the proven
  optima sum), every level boundary observed, zero protocol refusals.
- **Lean-simulator sweep**: 24/25 routes verified to WIN at exact
  optima; sb26's winning click is live-true (won at 124/124).
- **Egress audit**: the live run's complete egress = 5 pooled tunnels to
  three.arcprize.org, 0 blocked; a deliberate LLM-endpoint call is
  BLOCKED and logged.

## License

MIT (harness). The routebook and snapshot carry the competition's
open-data posture (CC0 on the Kaggle dataset).
