"""zAgent on ARC-AGI-3: the minimal faithful embodiment (docs/arc3-zagent.md).

A Python harness drives the zAgent mind (via the `zagent-sidecar` JSON-lines
seam) through ARC-AGI-3 episodes served by nars-memento's `Arc3Env`:

    frame-diff percepts ──▶ zmind_observe-shaped events  (the mind's :|: stream)
    op goals (once)     ──▶ <(*,{SELF}) --> ^op>!        (the body's tasking)
    act proposals       ──▶ env.step(ACTION*)           (the body performs)
    outcomes            ──▶ act-resolution / events     (the grounding loop)

The mind proposes acts; when it proposes none, a seeded-random fallback acts
so every step still feeds the evidence stream. Everything proposed, executed,
declined and observed is logged for the post-run analysis.
"""
