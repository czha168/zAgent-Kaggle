"""The truth-valued rulebook: Rule records and their workspace sidecar (§10).

The markdown rulebook stays the model-facing artifact (the revise/compile
sessions edit it exactly as before). The bridge maintains a parallel
`rules.json` — one `Rule` per behavioural rule — carrying:

- `text`  the natural-language statement (the carry-over key across edits)
- `nal`   an optional NAL statement, e.g. `<<(*,A,K) --> at> ==> <D --> open>>`
- `kind`  "fact" (established mechanic) or "hypothesis" (§10's default-
          negation open questions — seeded at low confidence)
- `pattern` optional attribution pattern over transition features
          (nars_bridge/evidence.py) — how replay evidence finds the rule
- `pos`/`neg` mechanical evidence counts from replay; truth derives from
          `seed.revise(from_evidence(pos, neg))` — LLM seeds are labelled
          judgment, mechanical counts are authoritative (Tier-A doctrine)

Sync preserves truth across rulebook revisions by normalized-text match, so
the LLM rewriting the markdown never resets accumulated evidence, and rule
ids stay stable for prompt references.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .truth import TruthValue, from_evidence

# Rules at or below this confidence are flagged "low-confidence" in the
# prompt sections — §10's "rules 3 and 7 are low-confidence" line.
LOW_CONFIDENCE = 0.5

# First N refuting transition ids kept on the record (AWM's EVIDENCE list:
# the falsifier stays visible after the revision that answers it). Recomputed
# by every rescan; never carried by sync.
FALSIFIED_BY_CAP = 8

# First N confirming transition ids kept on the record — the symmetric half
# of the EVIDENCE list: the reflect diagnosis "rule confirms here, engine
# diverges" (a code bug) reads differently from "rule refutes" (a claim bug)
# only when a sample of confirmers is visible beside the falsifiers.
# Recomputed by every rescan; never carried by sync.
CONFIRMED_BY_CAP = 8

# Withdrawal tombstones kept in rules.json (AWM's registry `fate`: rejection
# preservation extends to the hypothesis lifecycle). A rule dropped by a
# sync is recorded here so the mining harvest cannot resurrect it — without
# a tombstone, a deleted refuted statement is text-deduped against live
# records only and stages again on the next harvest. Newest kept, oldest
# dropped past the cap.
MAX_RETIREMENTS = 64

# The derived fates. "refuted" = the pattern fired but never confirmed
# (pos == 0 and neg > 0) — recomputed by every rescan, like the evidence
# lists; "active" otherwise (vacuous pos=neg=0 rules stay active: their
# emptiness is the census's problem, not fate's).
FATE_ACTIVE = "active"
FATE_REFUTED = "refuted"

SEED_FACT = TruthValue(1.0, 0.5)  # LLM-declared established mechanic
SEED_HYPOTHESIS = TruthValue(0.5, 0.1)  # default-negation open question


def normalize_text(text: str) -> str:
    return " ".join(text.split()).casefold()


@dataclass
class Rule:
    id: str
    text: str
    nal: str | None = None
    kind: str = "hypothesis"  # "fact" | "hypothesis"
    pattern: dict | None = None
    pos: int = 0
    neg: int = 0
    updated_iteration: int = 0
    # zmind_lsp parse verdicts for `nal` (nars_bridge/narsese.py):
    # None = never checked (server unavailable), [] = parses clean.
    nal_diagnostics: list[str] | None = None
    # ascending transition t values where the pattern was refuted (P2)
    falsified_by: list[int] = field(default_factory=list)
    # ascending transition t values where the pattern was confirmed (the
    # EVIDENCE list's positive half) — recomputed, never carried by sync
    confirmed_by: list[int] = field(default_factory=list)
    # derived lifecycle status (active | refuted), recomputed by every
    # rescan; never carried by sync
    fate: str = FATE_ACTIVE
    # one line: when this rule must NOT apply — the NA region, stated (P2)
    scope_note: str = ""

    @property
    def seed(self) -> TruthValue:
        return SEED_FACT if self.kind == "fact" else SEED_HYPOTHESIS

    @property
    def truth(self) -> TruthValue:
        """Declared seed revised by the mechanical replay evidence."""
        return self.seed.revise(from_evidence(self.pos, self.neg))

    def to_row(self) -> dict:
        return asdict(self)

    @classmethod
    def from_row(cls, row: dict) -> "Rule":
        return cls(
            id=str(row["id"]),
            text=str(row["text"]),
            nal=row.get("nal"),
            kind=row.get("kind", "hypothesis"),
            pattern=row.get("pattern"),
            pos=int(row.get("pos", 0)),
            neg=int(row.get("neg", 0)),
            updated_iteration=int(row.get("updated_iteration", 0)),
            nal_diagnostics=row.get("nal_diagnostics"),
            falsified_by=[int(t) for t in (row.get("falsified_by") or [])],
            confirmed_by=[int(t) for t in (row.get("confirmed_by") or [])],
            fate=str(row.get("fate", FATE_ACTIVE)),
            scope_note=str(row.get("scope_note", "")),
        )


class RuleStore:
    """`rules.json` sidecar in the run workspace."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.rules: list[Rule] = []
        self.retired: list[dict] = []  # withdrawal tombstones, newest last
        self.rulebook_sha: str | None = None
        self.last_sync_iteration: int | None = None
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.rules = [Rule.from_row(r) for r in data.get("rules", [])]
            self.retired = list(data.get("retired") or [])
            self.rulebook_sha = data.get("rulebook_sha")
            self.last_sync_iteration = data.get("last_sync_iteration")

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {
                    "rules": [r.to_row() for r in self.rules],
                    "retired": self.retired[-MAX_RETIREMENTS:],
                    "rulebook_sha": self.rulebook_sha,
                    "last_sync_iteration": self.last_sync_iteration,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def get(self, rule_id: str) -> Rule | None:
        return next((r for r in self.rules if r.id == rule_id), None)

    # -- sync ----------------------------------------------------------------

    def sync(self, records: list[Rule], iteration: int, rulebook_sha: str) -> dict:
        """Replace the record list, carrying evidence across the revision.

        Records whose normalized text matches an existing rule keep that
        rule's id, evidence counts, and updated_iteration; genuinely new
        statements get fresh ids. Dropped records are tombstoned in
        `retired` (newest last, capped) so downstream staging channels can
        refuse to resurrect them. Returns {kept, added, dropped}."""
        by_text = {normalize_text(r.text): r for r in self.rules}
        carried: list[Rule] = []
        used_ids = {r.id for r in self.rules}
        kept = 0
        dropped: list[Rule] = []
        for record in records:
            old = by_text.pop(normalize_text(record.text), None)
            if old is not None:
                kept += 1
                record.id = old.id
                record.pos = old.pos
                record.neg = old.neg
                record.updated_iteration = old.updated_iteration
                # the NA region rides the statement (falsified_by is recomputed
                # from the patterns, never carried)
                record.scope_note = old.scope_note
            else:
                record.id = record.id if record.id and record.id not in used_ids else self._next_id(carried, used_ids)
                used_ids.add(record.id)
            carried.append(record)
        dropped = list(by_text.values())
        for rule in dropped:
            self.retired.append(
                {
                    "id": rule.id,
                    "text": rule.text,
                    "nal": rule.nal,
                    "kind": rule.kind,
                    "pos": rule.pos,
                    "neg": rule.neg,
                    "fate": rule.fate,
                    "retired_iteration": iteration,
                }
            )
        result = {
            "kept": kept,
            "added": len(records) - kept,
            "dropped": len(dropped),
            "retired": len(dropped),
        }
        self.rules = carried
        self.rulebook_sha = rulebook_sha
        self.last_sync_iteration = iteration
        self.save()
        return result

    def _next_id(self, carried: list[Rule], used_ids: set[str]) -> str:
        numbers = set()
        for rule in [*self.rules, *carried]:
            if rule.id.startswith("R") and rule.id[1:].isdigit():
                numbers.add(int(rule.id[1:]))
        n = 1
        while n in numbers:
            n += 1
        rid = f"R{n}"
        used_ids.add(rid)
        return rid

    # -- confidence views -------------------------------------------------------

    def ranked_by_confidence(self) -> list[Rule]:
        """Least-confident first — the exploration-targeting order (§10)."""
        return sorted(self.rules, key=lambda r: (r.truth.c, r.id))

    def confidence_report(self) -> str:
        """Prompt section body: one bullet per rule, least confident first.

        Empty string when no records exist yet (callers then omit the
        section's content, not its header)."""
        if not self.rules:
            return ""
        lines = []
        for rule in self.ranked_by_confidence():
            tv = rule.truth
            if tv.c <= LOW_CONFIDENCE:
                flag = "low-confidence — prioritise experiments"
            elif tv.expectation() < 0.4:
                flag = "REFUTED by the record — revise or drop this rule"
            elif abs(tv.f - 0.5) < 0.2 and tv.c >= LOW_CONFIDENCE:
                flag = "split evidence — the rule needs refinement"
            else:
                flag = "confirmed"
            if rule.nal_diagnostics:  # the NAL formalization itself does not parse
                flag += " [nal: parse error]"
            evidence = f"evidence {rule.pos}+/{rule.neg}-"
            lines.append(
                f"- {rule.id} (f={tv.f:.2f}, c={tv.c:.2f}, {evidence}): "
                f"{rule.text} [{flag}]"
            )
        return "\n".join(lines)

    def lowest_confidence_ids(self, n: int = 2) -> list[str]:
        """§10's example output shape: ['R3', 'R7'] — the rules to test next."""
        return [r.id for r in self.ranked_by_confidence()[:n]]
