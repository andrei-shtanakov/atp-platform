# Design: open-prose receipts as an evaluation input — vendored contract + `receipt_chain` checker

**Date:** 2026-07-16
**Status:** Approved (brainstorm) — ready for implementation plan
**Origin:** open-prose contracts offer (handoff note
`2026-07-16-openprose-contracts-offer.md` in the dev-only KB sibling; pointer, not a link —
see CLAUDE.md on cross-repo references). Decision log: TODO.md item
"open-prose receipts/IR как evaluation-вход" (ACTIVE, 2026-07-16).
**Scope profile:** v1 is **receipts-only**. The IR contract is vendored (the two contracts
cross-reference: statement-id and canonical form live in `receipt.md`), but no IR reader or
IR validation ships in v1.

---

## Problem

open-prose runs now leave two machine-readable, keyless-verifiable artifacts per run:
`receipts.jsonl` (`openprose.receipt.v1` — an append-only, hash-chained ledger: one receipt
per completed statement, input/output fingerprints, honest token attribution via
`usage.basis: exact|estimated|unavailable`) and `{program}.ir.json`
(`openprose.compile-ir.v1`). ATP has no way to consume them: grading an open-prose run today
would mean parsing the human-readable `state.md` narrative.

ATP already owns the exact mechanics this needs — vendored pinned contracts with contract
tests in CI (`method/contract/learning-event-v1.schema.json`, RD-007 M1a) and a closed
deterministic-checker registry (`atp/evaluators/checkers/`, selected via
`grader: {type: programmatic, checker: <name>}`). This design lands the receipts contract on
those two existing surfaces. It also makes ATP the named consumer that open-prose's Rust
gate 4.6 ("a `receipts-verify` crate used by atp-platform", their plan, revisited at the end
of their Phase 4) predicates on — causality is correct only in this direction: working
reader first, gate trigger second.

**Non-goals (v1):** IR reader/validation; receipts → EvidenceRef/LearningEvent mapping;
usage rollup into the cost view (ADR-003d); verifying `bindings/*` files against
`output_fingerprint` on skipped receipts. All are follow-ups listed at the end.

## File layout

```
method/contract/openprose/
├── PROVENANCE.md            # source repo + commit (a0395cd), per-file sha256, vendoring date
├── receipt.md               # pinned copy of open-prose/contracts/receipt.md
└── ir.md                    # pinned copy of open-prose/contracts/ir.md (contract only, no reader)

method/contract/fixtures/openprose/
├── runs/                    # pinned copies of the 4 committed corpus runs
│   └── <run_id>/            #   receipts.jsonl, run.json, program.prose, bindings/, state.md
├── broken/                  # pinned copies of the corrupted-run fixtures, each with its
│   ├── broken-chain/        #   original expected.json ({ok, error_contains|warning_contains})
│   ├── tampered-content/
│   ├── torn-write/
│   └── truncated-ledger/
└── ir/                      # pinned IR fixtures (stale-source, tampered-ir, unknown-agent)
                             #   vendored for the future IR reader; NOT wired to any test in v1

atp/evaluators/openprose_receipts/
├── __init__.py
├── canonical.py             # canonical_json(), content_hash() — our implementation, stdlib
├── reader.py                # load_ledger(), verify_run() → VerifyResult
└── checker.py               # receipt_chain_check(config, text) → CaseVerdict

tests/contract/test_openprose_receipts_contract.py
tests/unit/evaluators/test_openprose_receipts.py
```

Vendoring is mandatory, not a convenience: CI does not have the polyrepo sibling, and
shipped/test code must never resolve paths outside the repo (workspace rule). Fixtures total
≈180 KB. `PROVENANCE.md` records the open-prose commit and a sha256 per vendored file so
drift against upstream is checkable by hand (byte-conformance automation is out of scope,
same posture as RD-007 M2).

## Reader (`atp/evaluators/openprose_receipts/`)

Private library. Logic is stdlib-only (`json`, `hashlib`); result models are pydantic, per
repo style. No `jsonschema`: the contract has no published `.schema.json`, structural checks
are plain code, and everything that actually decides the outcome (canonical form, chain,
anchor) is inexpressible in JSON Schema anyway. Zero new dependencies.

### `canonical.py`

- `canonical_json(value) -> bytes` — the contract's canonical form: keys sorted by byte
  order, no whitespace, UTF-8 strings **not** ASCII-escaped, `true/false/null` literals.
  Rejects (raises) floats, NaN, and Infinity — non-integers break hash portability and are
  invalid in a receipt by contract.
- `content_hash(receipt: dict) -> str` — `"sha256:" + hex(sha256(canonical(receipt sans
  content_hash)))`. The hash covers every field except `content_hash` itself, including
  `prev` and any unknown fields (append-frozen compatibility: unknown fields are ignored
  semantically but hashed as received).

Written against the vendored contract text; `open-prose/tools/.../canonical.py` was read as
a reference implementation, not copied.

### `reader.py`

```python
def load_ledger(path: Path) -> LoadedLedger   # parsed receipts + line-level issues
def verify_run(run_dir: Path) -> VerifyResult
```

`load_ledger` never raises on bad content — it accumulates line-level issues and returns
the parseable prefix, because "invalid JSON on the final line is a warning, prefix still
verifies" is impossible with a throwing loader.

```python
class Issue(BaseModel):
    code: str                 # stable machine code, see table
    line_no: int | None       # 1-based ledger line, None for run-level issues
    message: str              # human text; phrased to contain the upstream
                              # expected.json substrings verbatim

class VerifyResult(BaseModel):
    ok: bool                  # errors == []
    receipt_count: int        # receipts covered by verification — NOT physical
                              # line count: the parsed prefix on a torn line,
                              # the anchored prefix on a torn manifest
    errors: list[Issue]
    warnings: list[Issue]
```

`verify_run` checks, in order:

1. **Version** — every receipt's `v` must be `"openprose.receipt.v1"`; unknown `v` →
   error `unknown_version` (contract: consumers MUST refuse unknown versions).
2. **Structure** (in code, not jsonschema) — required fields present; `kind` / `status` /
   `usage.basis` enums valid; all numbers integers (float → error `invalid_number`);
   unknown fields ignored.
3. **Content hash** — recomputed over canonical form → error `content_hash_mismatch`
   (message contains "content_hash mismatch").
4. **Chain** — line N's `prev` equals line N−1's `content_hash`, line 1 has `prev: null` →
   error `chain_break` (message contains "prev broken").
5. **Anchor** — `run.json` (`openprose.run.v1`): `ledger_head` must equal the last
   receipt's `content_hash` → error `ledger_head_mismatch` on divergence (message contains
   "ledger_head") — **except** the torn-write manifest case below. With a matching head,
   `run.json.receipt_count` must equal the ledger's receipt count → error
   `receipt_count_mismatch`. Missing `run.json` → warning `no_anchor`, chain-only
   verification (the contract explicitly allows distinguishing interrupted runs without
   the manifest). An empty `receipts.jsonl` (a run always opens with a `run_start`
   control receipt) → error `empty_ledger`.

### Torn-write: two distinct cases, both warnings

A torn write is a crash mid-append, not tampering; the valid prefix stands. Upstream's
`expected.json` semantics (`ok: true`, `warning_contains: "torn write"`) are preserved.

- **`torn_write_manifest`** — every ledger line is valid, but `run.json` anchors an
  *earlier* receipt: `ledger_head` equals the `content_hash` of receipt K < N and
  `receipt_count == K` ("append succeeded, head update did not"). This is the upstream
  `torn-write` fixture. Warning; the anchored prefix is verified; `receipt_count` in the
  result is K (the anchored prefix), matching upstream semantics. If `ledger_head` matches
  *no* receipt in the ledger, it is a plain `ledger_head_mismatch` error.
- **`torn_write_line`** — the **final** physical line is not valid JSON (partial write).
  Warning; the parsed prefix is verified; `receipt_count` counts the prefix. Invalid JSON
  on any *non-final* line is an error `invalid_json` — only a trailing torn line is
  excusable as a crash artifact. Upstream has no fixture for this case; we cover it with
  our own unit fixture.

`ok` stays `True` when the only findings are warnings.

## Checker `receipt_chain`

Registered fourth in `atp/evaluators/checkers/__init__.py`. Selected as:

```yaml
grader:
  type: programmatic
  checker: receipt_chain
  config:
    run_dir: runs/my-run        # relative to the case directory, mandatory
```

Signature is the registry's `Checker = Callable[[dict, str | None], CaseVerdict]`. The
`text` argument is ignored — the artifact under test is a run *directory*, not agent stdout.

### `_case_dir` injection (dispatch-layer change)

The checker signature carries no notion of "the case's directory", so the confinement root
must be injected by the dispatch layer — this is the one change outside the new package:

- `AgentEvalCaseEvaluator._evaluate_critical()` (`packages/atp-method/atp_method/
  evaluators/case_evaluator.py`) builds the checker config as a **copy** of
  `assertion.config` and **unconditionally** sets `_case_dir` from
  `task.input_data["case_path"]`'s parent directory (same derivation as
  `corpus.py`). Injection is generic — every checker gets it; existing checkers ignore
  unknown config keys.
- A user-supplied `_case_dir` in `grader.config` is always overwritten (never trusted) —
  the underscore prefix marks the key as dispatch-owned.
- If `case_path` is absent from `input_data`, `_case_dir` is not injected.

### Path confinement and verdict mapping

- `run_dir` must be present, **relative** (absolute paths are rejected in
  evaluation-case mode), and must resolve — after `(_case_dir / run_dir).resolve()` —
  to a path inside `_case_dir`. No `_case_dir` in config → any `run_dir` is unresolvable.
- Verdict mapping keeps "not gradeable" and "failed verification" distinct:

| Condition | Verdict |
| --- | --- |
| `run_dir` missing/absolute/escaping confinement, `_case_dir` absent, directory missing or unreadable, `receipts.jsonl` absent | `malformed=True`, `critical_pass=False` |
| `verify_run` ran | `malformed=False`, `critical_pass=result.ok`, `rubric_score=1.0 if ok else 0.0` |

- `details` carries `{"errors": [...], "warnings": [...], "receipt_count": N}` (issues as
  dicts). `grader_version = "receipt_chain@1"`. `recall`/`precision`/`fp_count` keep their
  defaults — they are findings-checker vocabulary and do not apply.

## Testing

**Contract tests** (`tests/contract/test_openprose_receipts_contract.py`), over the
vendored fixtures only:

- Each of the 4 corpus runs: `verify_run` → `ok=True`, no errors, no warnings,
  `receipt_count` matches `run.json`.
- Each broken fixture: outcome asserted **through its vendored `expected.json`** —
  `ok` matches; `error_contains`/`warning_contains` substring found in the corresponding
  issue messages. This keeps upstream's fixtures consumable verbatim and makes future
  fixture refreshes mechanical.

**Unit tests** (`tests/unit/evaluators/test_openprose_receipts.py`):

- Canonicalization: key sorting, UTF-8 not escaped, nested containers, float/NaN/Infinity
  rejection, unknown fields included in the hash.
- `verify_run` edges: empty ledger (`empty_ledger`), unknown `v`, `receipt_count`
  mismatch with a matching head (`receipt_count_mismatch`),
  missing `run.json` (warning `no_anchor`), `torn_write_line` (own minimal fixture,
  distinct from the vendored `torn_write_manifest` case), invalid JSON on a non-final
  line (error).
- Checker: verdict mapping table above; `_case_dir` injection (evaluator overwrites a
  user-supplied `_case_dir`; missing `case_path` → no injection → relative `run_dir` is
  malformed); absolute `run_dir` rejected; `../` escape rejected.

## Follow-ups (explicitly out of v1)

- IR reader/validation (`ir-check` semantics); the contract and broken fixtures are
  already vendored.
- receipts → EvidenceRef / LearningEvent mapping (arbiter-facing evidence chain).
- Usage rollup into the cost view: receipt `usage.basis` is a sibling of
  `cloud_pricing_usage_v1` semantics (ADR-003d) — a receipts-sourced cost view would be
  derived-not-stored, same as `method/price_reports.py`.
- `bindings/*` content verification against `output_fingerprint` (incl. skipped receipts'
  copied-forward bindings).
- Automated byte-conformance check of the vendored copies against upstream (manual via
  `PROVENANCE.md` sha256 until then; same posture as RD-007 M2).
